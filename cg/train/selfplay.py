"""Batched AlphaZero-style self-play for uttt.ai's network under CodinGame rules.

Search is uttt.ai's NMCTS (Q + c * max(0.01, P) * sqrt(N) / (n + 1), c = 2.0,
backed-up means, tree reuse between moves). Each worker process runs many games
in lockstep: every round advances each game's search to one leaf, evaluates all
leaves in one GPU batch, then backs the values up. Child positions are built
only when first visited. Root priors get Dirichlet noise; the first
TEMPERATURE_PLIES moves are sampled in proportion to visits, later moves take
the most-visited child.

Records per move: state (93 bytes), visit distribution over 81 squares, root
value (side to move), then the final result under CodinGame rules.

usage: selfplay.py <net.pt | onnx> <out_dir> <games> <sims> <workers> [games_per_worker_batch]
                   [--backend torch|ort] [--fp16] [--cap P --cheap-sims N] [--bench SECONDS]

--backend ort   run the network with onnxruntime's DirectML provider (the .onnx next to a .pt)
--fp16          half-precision inference (torch backend)
--cap P         playout-cap randomization: a move gets the full <sims> search with
                probability P and is recorded; otherwise a --cheap-sims search, not recorded
--bench S       measure for S seconds after a 30 s warm-up, report throughput and a
                per-worker time profile, save nothing
"""
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # this repository's utttpy (CodinGame rules)
from utttpy.game.action import Action  # noqa: E402
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe  # noqa: E402

C_PUCT = 2.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25
TEMPERATURE_PLIES = 12

# state index (miniboard * 9 + square) -> row, col on the 9x9 grid
ROW = np.array([3 * (s // 27) + (s % 9) // 3 for s in range(81)])
COL = np.array([3 * ((s // 9) % 3) + s % 3 for s in range(81)])


def encode(u: UltimateTicTacToe, legal) -> np.ndarray:
    """uttt.ai's 4x9x9 input (get_state_ndarray_4x9x9), vectorised."""
    cells = np.frombuffer(bytes(u.state[:81]), dtype=np.uint8)
    me, opp = (1, 2) if u.next_symbol == 1 else (2, 1)
    x = np.zeros((4, 9, 9), dtype=np.float32)
    m = cells == me
    x[0, ROW[m], COL[m]] = 1
    m = cells == opp
    x[1, ROW[m], COL[m]] = 1
    x[2] = 1.0 if u.next_symbol == 1 else -1.0
    idx = np.array(legal, dtype=np.int64)
    x[3, ROW[idx], COL[idx]] = 1
    return x


class Node:
    __slots__ = ("u", "action", "prior", "children", "n", "w", "legal")

    def __init__(self, u, action=None, prior=0.0):
        self.u = u            # built lazily for children
        self.action = action  # state index of the move that led here
        self.prior = prior
        self.children = None  # list after expansion
        self.n = 0
        self.w = 0.0          # sum of values from this node's side to move
        self.legal = None


def terminal_value(u):
    """Value for the side to move at a finished position (CodinGame rules via the fork)."""
    if u.is_result_draw():
        return 0.0
    return 1.0 if u.result == u.next_symbol else -1.0


class Game:
    def __init__(self, rng, sims, cap_p=1.0, cheap_sims=None):
        self.rng = rng
        self.full_sims = sims
        self.cap_p = cap_p
        self.cheap_sims = cheap_sims or sims
        self._choose_budget()
        self.u = UltimateTicTacToe()
        self.root = Node(self.u.clone())
        self.records = []  # (state bytes, visits[81], root value, side to move)
        self.ply = 0
        self.pending = None  # (path, leaf) awaiting evaluation
        self.done = False

    def _choose_budget(self):
        # No random draw when the cap is off, so default runs keep their random stream.
        self.full_now = self.cap_p >= 1.0 or self.rng.random() < self.cap_p
        self.sims = self.full_sims if self.full_now else self.cheap_sims

    def select(self):
        """Walk to a leaf. Terminal leaves are backed up immediately (returns None)."""
        node = self.root
        path = [node]
        while node.children is not None and node.children:
            sq = math.sqrt(node.n)
            best, best_s = None, -1e9
            for c in node.children:
                q = -(c.w / c.n) if c.n else 0.0
                s = q + C_PUCT * max(0.01, c.prior) * sq / (c.n + 1)
                if s > best_s:
                    best, best_s = c, s
            node = best
            if node.u is None:
                u = path[-1].u.clone()
                u.execute(Action(symbol=u.next_symbol, index=node.action), verify=False)
                node.u = u
            path.append(node)
        if node.u.is_terminated():
            self.backup(path, terminal_value(node.u))
            return None
        if node.legal is None:
            node.legal = node.u.get_legal_indexes()
        self.pending = path
        return node

    def backup(self, path, v):
        for node in reversed(path):
            node.n += 1
            node.w += v
            v = -v

    def expand_and_backup(self, logits, value):
        path = self.pending
        leaf = path[-1]
        legal = leaf.legal
        picked = logits[ROW[legal], COL[legal]].astype(np.float64)
        p = np.exp(picked - picked.max())
        p /= p.sum()
        if leaf is self.root and self.root.children is None and self.full_now:
            p = self._noisy(p)
        leaf.children = [Node(None, a, float(pr)) for a, pr in zip(legal, p)]
        self.pending = None
        self.backup(path, float(value))

    def _noisy(self, p):
        noise = np.random.default_rng(self.rng.getrandbits(32)).dirichlet([DIRICHLET_ALPHA] * len(p))
        return (1 - DIRICHLET_EPS) * p + DIRICHLET_EPS * noise

    def search_done(self):
        return self.root.n >= self.sims

    def play_move(self):
        kids = self.root.children
        visits = np.array([c.n for c in kids], dtype=np.float64)
        dist = np.zeros(81, dtype=np.float32)
        for c, v in zip(kids, visits):
            dist[c.action] = v
        dist /= dist.sum()
        if self.full_now:
            self.records.append((bytes(self.u.state), dist, self.root.w / max(1, self.root.n), self.u.next_symbol))
        if self.ply < TEMPERATURE_PLIES:
            child = self.rng.choices(kids, weights=visits)[0]
        else:
            top = visits.max()
            child = self.rng.choice([c for c, v in zip(kids, visits) if v >= top])
        self.u.execute(Action(symbol=self.u.next_symbol, index=child.action), verify=False)
        self.ply += 1
        if self.u.is_terminated():
            self.done = True
            return
        if child.u is None:
            child.u = self.u.clone()
        self.root = child  # tree reuse
        self._choose_budget()
        if self.root.children is not None and self.full_now:  # fresh noise on the reused root's priors
            p = np.array([c.prior for c in self.root.children])
            for c, q in zip(self.root.children, self._noisy(p / p.sum())):
                c.prior = float(q)

    def results(self):
        """Training rows: state, policy, root value, outcome for the side to move."""
        rows = []
        for state, dist, q, stm in self.records:
            z = 0.0 if self.u.is_result_draw() else (1.0 if self.u.result == stm else -1.0)
            rows.append((state, dist, q, z))
        return rows


class Evaluator:
    def __init__(self, net_path, backend="torch", fp16=False):
        self.backend = backend
        if backend == "ort":
            import onnxruntime as ort
            import onnx_weights
            onnx_path = str(net_path) if str(net_path).endswith(".onnx") else str(net_path)[:-3] + ".onnx"
            tmp = HERE / f"_dynbatch_{os.getpid()}.onnx"  # the shipped graph fixes batch = 1
            onnx_weights.set_dynamic_batch(onnx_path, tmp)
            opts = ort.SessionOptions()
            opts.enable_mem_pattern = False  # required by the DirectML provider
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.sess = ort.InferenceSession(str(tmp), sess_options=opts, providers=["DmlExecutionProvider"])
            tmp.unlink()
            return
        import torch
        import torch_directml
        import fused_net
        self.torch = torch
        self.fp16 = fp16
        self.dev = torch_directml.device()
        if str(net_path).endswith(".onnx"):
            net = fused_net.from_onnx(net_path)
        else:
            net = fused_net.FusedPolicyValueNet()
            net.load_state_dict(torch.load(net_path, map_location="cpu", weights_only=True))
        net = net.to(self.dev).eval()
        self.net = net.half() if fp16 else net

    def __call__(self, xs):
        x = np.stack(xs)
        if self.backend == "ort":
            return tuple(self.sess.run(["policy_logits", "state_value"], {"input": x}))
        with self.torch.no_grad():
            t = self.torch.from_numpy(x).to(self.dev)
            pl, v = self.net(t.half() if self.fp16 else t)
            return pl.float().cpu().numpy(), v.float().cpu().numpy()


def worker(wid, net_path, out_dir, n_games, sims, batch_games, seed, progress, opts):
    import warnings
    warnings.filterwarnings("ignore")
    rng = random.Random(seed)
    ev = Evaluator(net_path, opts["backend"], opts["fp16"])

    def new_game():
        return Game(random.Random(rng.getrandbits(64)), sims, opts["cap"], opts["cheap_sims"])

    bench = opts["bench"]
    if bench:
        n_games = 10 ** 9
    games = [new_game() for _ in range(min(batch_games, n_games))]
    started = len(games)
    rows = []
    shard = 0
    # Profile buckets and counters; reset when the bench warm-up ends.
    keys = ("select", "encode", "eval", "backup", "move")
    tm = dict.fromkeys(keys, 0.0)
    ct = dict(evals=0, batches=0, moves=0, positions=0, games=0)
    clock = time.perf_counter
    warm_until = clock() + 30
    measuring = not bench
    t_measure = clock()
    while games:
        now = clock()
        if bench and not measuring and now >= warm_until:
            tm = dict.fromkeys(keys, 0.0)
            ct = dict.fromkeys(ct, 0)
            measuring, t_measure = True, now
        if bench and measuring and now - t_measure >= bench:
            progress.put(("stats", wid, now - t_measure, tm, ct))
            return
        leaves, owners = [], []
        t0 = clock()
        enc = 0.0
        for g in games:
            for _ in range(4):  # terminal leaves back up for free; try a few selections
                leaf = g.select()
                if leaf is not None:
                    te = clock()
                    leaves.append(encode(leaf.u, leaf.legal))
                    enc += clock() - te
                    owners.append(g)
                    break
        t1 = clock()
        tm["select"] += t1 - t0 - enc
        tm["encode"] += enc
        if leaves:
            logits, values = ev(leaves)
            t2 = clock()
            tm["eval"] += t2 - t1
            ct["evals"] += len(leaves)
            ct["batches"] += 1
            for g, lg, v in zip(owners, logits, values):
                g.expand_and_backup(lg, v)
            t1 = clock()
            tm["backup"] += t1 - t2
        nxt = []
        for g in games:
            if g.pending is None and g.search_done():
                ct["positions"] += g.full_now
                g.play_move()
                ct["moves"] += 1
            if g.done:
                if not bench:
                    rows += g.results()
                    progress.put(1)
                ct["games"] += 1
                if started < n_games:
                    nxt.append(new_game())
                    started += 1
            else:
                nxt.append(g)
        games = nxt
        tm["move"] += clock() - t1
        if not bench and (len(rows) >= 20000 or (not games and rows)):
            save(rows, Path(out_dir) / f"w{wid:02d}_{shard:03d}.npz")
            rows, shard = [], shard + 1


def save(rows, path):
    states = np.frombuffer(b"".join(r[0] for r in rows), dtype=np.uint8).reshape(len(rows), 93)
    np.savez_compressed(path, state=states, policy=np.stack([r[1] for r in rows]).astype(np.float16),
                        q=np.array([r[2] for r in rows], dtype=np.float32),
                        z=np.array([r[3] for r in rows], dtype=np.float32))


def parse_opts(argv):
    opts = dict(backend="torch", fp16=False, cap=1.0, cheap_sims=None, bench=0)
    pos, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--backend":
            opts["backend"] = argv[i + 1]
            i += 2
        elif a == "--fp16":
            opts["fp16"] = True
            i += 1
        elif a == "--cap":
            opts["cap"] = float(argv[i + 1])
            i += 2
        elif a == "--cheap-sims":
            opts["cheap_sims"] = int(argv[i + 1])
            i += 2
        elif a == "--bench":
            opts["bench"] = float(argv[i + 1])
            i += 2
        else:
            pos.append(a)
            i += 1
    return pos, opts


def report_bench(stats, workers, opts, sims, batch_games):
    secs = sum(st[2] for st in stats) / workers
    tot = {k: sum(st[4][k] for st in stats) for k in stats[0][4]}
    tm = {k: sum(st[3][k] for st in stats) / workers for k in stats[0][3]}
    print(f"BENCH backend={opts['backend']} fp16={opts['fp16']} workers={workers} games/worker={batch_games} "
          f"sims={sims} cap={opts['cap']} cheap={opts['cheap_sims']}")
    print(f"  over {secs:.0f} s: {tot['evals'] / secs:,.0f} evals/s, mean batch {tot['evals'] / max(1, tot['batches']):.0f}, "
          f"{tot['moves'] / secs * 3600:,.0f} moves/h, {tot['positions'] / secs * 3600:,.0f} recorded positions/h, "
          f"{tot['games'] / secs * 3600:,.0f} games/h")
    busy = sum(tm.values())
    print("  per-worker time: " + ", ".join(f"{k} {v / secs:.0%}" for k, v in tm.items())
          + f", unaccounted {1 - busy / secs:.0%}", flush=True)


def main():
    import multiprocessing as mp
    pos, opts = parse_opts(sys.argv[1:])
    net_path, out_dir, n_games, sims, workers = pos[0], pos[1], int(pos[2]), int(pos[3]), int(pos[4])
    batch_games = int(pos[5]) if len(pos) > 5 else 128
    if not opts["bench"]:
        os.makedirs(out_dir, exist_ok=True)
    progress = mp.Queue()
    per = [n_games // workers + (1 if i < n_games % workers else 0) for i in range(workers)]
    procs = [mp.Process(target=worker, args=(i, net_path, out_dir, per[i], sims, batch_games, 1000 + i, progress, opts))
             for i in range(workers)]
    for p in procs:
        p.start()
    if opts["bench"]:
        stats = [progress.get() for _ in range(workers)]
        for p in procs:
            p.join()
        report_bench(stats, workers, opts, sims, batch_games)
        return
    t0 = time.time()
    for k in range(1, n_games + 1):
        progress.get()
        if k % 50 == 0 or k == n_games:
            el = time.time() - t0
            eta = time.strftime("%H:%M", time.localtime(t0 + el / k * n_games))
            print(f"[{time.strftime('%H:%M:%S')}] {k}/{n_games} games  {k / el * 3600:,.0f} games/h  ETA {eta}", flush=True)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
