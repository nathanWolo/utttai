"""Grow crossfish's CodinGame opening book with uttt.ai.

The book is a tree after the first player's center-center (crossfish always
opens there, and the book assumes an opponent who moves first does too). It is
grown best-first by the estimated chance of reaching each line:

  - opponent positions: the replies uttt.ai's network gives prior >= --cover
    (0.03) are covered; the rest leave the book. Each covered reply carries on
    the line's reach times q = 0.5 * prior / (covered prior) + 0.5 / |covered|,
    hedging toward a flatter opponent than uttt.ai's own policy.
  - our positions: uttt.ai's move after a --sims (3,200) simulation search on the
    GPU, unless crossfish vetoes it: crossfish searches --cf-ms (500) and, if it
    prefers another move and scores uttt.ai's more than --veto (500) worse, its
    own move is kept.

Opponent positions are expanded in order of reach until the payload budget
(--chars of 14-bit characters) is spent. Costs follow the packed format of
cpp_impl/play_book.hpp: our move's index plus a "continues" digit at each of
our positions, and one "covered" digit per non-terminal reply at each
expanded opponent position.

Output: one "S <seq> <move>" line per of our positions, for
cpp_impl/play_book_pack (which replays the lines and merges symmetric and
transposed positions), plus a progress log.

usage: python uttt_book_gen.py <out.txt> --cf <crossfish_debug.exe> [--net train/net4.onnx]
           [--chars 16000] [--cover 0.03] [--sims 3200] [--cf-ms 500] [--veto 500]
           [--gpu-workers 6] [--cf-workers 7] [--round 256]
"""
import argparse
import heapq
import math
import multiprocessing as mp
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CG = HERE.parent
sys.path.insert(0, str(CG / "train"))
sys.path.insert(0, str(CG.parent))
from utttpy.game.action import Action  # noqa: E402
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe  # noqa: E402

CENTER = 40


def game(seq):
    u = UltimateTicTacToe()
    for i in seq:
        u.execute(Action(symbol=u.next_symbol, index=i), verify=False)
    return u


# ---------------------------------------------------------------- symmetry (dedupe only; the packer merges exactly)
def _sym_rc(t, n, r, c):
    m = n - 1
    return [(r, c), (c, m - r), (m - r, m - c), (m - c, r), (r, m - c), (m - r, c), (c, r), (m - c, m - r)][t]


SYM_CELL = []
for _t in range(8):
    row = []
    for idx in range(81):
        mb, sq = divmod(idx, 9)
        r, c = (mb // 3) * 3 + sq // 3, (mb % 3) * 3 + sq % 3
        orow, ocol = _sym_rc(_t, 9, r, c)
        row.append(((orow // 3) * 3 + ocol // 3) * 9 + (orow % 3) * 3 + ocol % 3)
    SYM_CELL.append(row)
SYM_BOARD = [[_sym_rc(t, 3, b // 3, b % 3)[0] * 3 + _sym_rc(t, 3, b // 3, b % 3)[1] for b in range(9)] + [9]
             for t in range(8)]


def canon(u):
    """Symmetry-canonical key of a position (cells and the board to play)."""
    legal = u.get_legal_indexes()
    boards = {i // 9 for i in legal}
    active = boards.pop() if len(boards) == 1 else 9
    cells = u.state[:81]
    best = None
    for t in range(8):
        k = [0] * 81
        for i in range(81):
            k[SYM_CELL[t][i]] = cells[i]
        key = bytes(k) + bytes([SYM_BOARD[t][active]])
        if best is None or key < best:
            best = key
    return best


# ---------------------------------------------------------------- GPU workers: priors and deep searches
def gpu_worker(net, jobs, results):
    import warnings
    warnings.filterwarnings("ignore")
    import selfplay
    from netmatch import Side
    ev = selfplay.Evaluator(net, "ort")
    while True:
        job = jobs.get()
        if job is None:
            return
        kind, tag, seqs, sims = job
        if kind == "prior":
            xs, legals = [], []
            for s in seqs:
                u = game(s)
                legal = u.get_legal_indexes()
                legals.append(legal)
                xs.append(selfplay.encode(u, legal))
            logits, _ = ev(xs)
            out = []
            for lg, legal in zip(logits, legals):
                picked = lg[selfplay.ROW[legal], selfplay.COL[legal]].astype(np.float64)
                p = np.exp(picked - picked.max())
                p /= p.sum()
                out.append(dict(zip(legal, p.tolist())))
            results.put((tag, out))
        else:  # deep search, all positions of the chunk searched together in one batch per round
            sides = [Side(game(s), sims, random.Random(hash(tuple(s)) & 0xFFFF)) for s in seqs]
            live = list(range(len(sides)))
            while live:
                leaves, owners = [], []
                for i in live:
                    g = sides[i]
                    for _ in range(4):
                        leaf = g.select()
                        if leaf is not None:
                            leaves.append(selfplay.encode(leaf.u, leaf.legal))
                            owners.append(g)
                            break
                if leaves:
                    logits, values = ev(leaves)
                    for g, lg, v in zip(owners, logits, values):
                        g.expand_and_backup(lg, v)
                live = [i for i in live if not (sides[i].pending is None and sides[i].search_done())]
            out = []
            for g in sides:
                kids = g.root.children
                top = max(c.n for c in kids)
                best = min((c for c in kids if c.n >= top), key=lambda c: c.action)
                out.append((best.action, g.root.w / max(1, g.root.n)))
            results.put((tag, out))


class Gpu:
    def __init__(self, net, workers):
        self.jobs, self.results = mp.Queue(), mp.Queue()
        self.procs = [mp.Process(target=gpu_worker, args=(net, self.jobs, self.results), daemon=True)
                      for _ in range(workers)]
        for p in self.procs:
            p.start()
        self.n = workers

    def run(self, kind, seqs, sims=0):
        if not seqs:
            return []
        size = max(1, math.ceil(len(seqs) / self.n))
        chunks = [seqs[i:i + size] for i in range(0, len(seqs), size)]
        for tag, c in enumerate(chunks):
            self.jobs.put((kind, tag, c, sims))
        got = dict(self.results.get() for _ in chunks)
        return [x for tag in range(len(chunks)) for x in got[tag]]

    def close(self):
        for _ in self.procs:
            self.jobs.put(None)


# ---------------------------------------------------------------- crossfish veto
class Crossfish:
    def __init__(self, exe, workers, ms, veto):
        self.exe, self.ms, self.veto = exe, ms, veto
        self.pool = queue.Queue()
        for _ in range(workers):
            self.pool.put(subprocess.Popen([exe, "match"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, text=True, bufsize=1))

    def think(self, proc, seq):
        proc.stdin.write("NEW\n" + "".join(f"APPLY {i // 9} {i % 9}\n" for i in seq) + f"THINK {self.ms}\n")
        proc.stdin.flush()
        while True:
            parts = proc.stdout.readline().split()
            if parts and parts[0] != "READY":
                return int(parts[0]) * 9 + int(parts[1]), int(parts[3])

    def check(self, seq, ut_move):
        """(move to book, vetoed?)"""
        proc = self.pool.get()
        try:
            cf_move, cf_score = self.think(proc, seq)
            if cf_move == ut_move:
                return ut_move, False
            u = game(seq + [ut_move])
            if u.is_terminated():  # uttt.ai's move ends the game: keep it unless it loses
                return (ut_move, False) if not (u.result and u.result != game(seq).next_symbol) else (cf_move, True)
            _, reply_score = self.think(proc, seq + [ut_move])
            if cf_score - (-reply_score) > self.veto:
                return cf_move, True
            return ut_move, False
        finally:
            self.pool.put(proc)

    def check_all(self, items, workers):
        out = [None] * len(items)
        it = iter(enumerate(items))
        lock = threading.Lock()

        def run():
            while True:
                with lock:
                    nxt = next(it, None)
                if nxt is None:
                    return
                k, (seq, mv) = nxt
                out[k] = self.check(seq, mv)
        threads = [threading.Thread(target=run) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return out


# ---------------------------------------------------------------- growing the tree
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cf", required=True)
    ap.add_argument("--net", default=str(CG / "train" / "net4.onnx"))
    ap.add_argument("--chars", type=int, default=16000)
    ap.add_argument("--cover", type=float, default=0.03)
    ap.add_argument("--sims", type=int, default=3200)
    ap.add_argument("--cf-ms", type=int, default=500)
    ap.add_argument("--veto", type=int, default=500)
    ap.add_argument("--gpu-workers", type=int, default=6)
    ap.add_argument("--cf-workers", type=int, default=7)
    ap.add_argument("--round", type=int, default=256, help="opponent positions expanded per round")
    a = ap.parse_args()

    budget = a.chars * 14
    gpu = Gpu(a.net, a.gpu_workers)
    cf = Crossfish(a.cf, a.cf_workers, a.cf_ms, a.veto)
    ours = {}            # canon key -> (seq, move)
    seen_opp = set()
    bits = 0.0
    vetoes = 0
    frontier = []        # (-reach, tiebreak, seq) of opponent positions not yet expanded
    tick = 0
    log = open(Path(a.out).with_suffix(".log"), "w", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def add_ours(items):
        """items: (seq, reach) of our positions. Search, veto, record, queue what follows."""
        nonlocal bits, vetoes, tick
        fresh = []
        for seq, reach in items:
            k = canon(game(seq))
            if k not in ours:
                ours[k] = None
                fresh.append((seq, reach, k))
        if not fresh:
            return
        found = gpu.run("search", [s for s, _, _ in fresh], a.sims)
        decided = cf.check_all([(s, mv) for (s, _, _), (mv, _) in zip(fresh, found)], a.cf_workers)
        for (seq, reach, k), (mv, vetoed) in zip(fresh, decided):
            vetoes += vetoed
            ours[k] = (seq, mv)
            bits += math.log2(len(game(seq).get_legal_indexes())) + 1  # move index + "continues"
            nxt = seq + [mv]
            if not game(nxt).is_terminated():
                tick += 1
                heapq.heappush(frontier, (-reach, tick, nxt))

    t0 = time.time()
    add_ours([([CENTER], 1.0)])                    # we move second, after the opponent's center-center
    heapq.heappush(frontier, (-1.0, 0, [CENTER]))  # we move first: the bot opened center-center
    rounds = 0
    while frontier and bits < budget:
        batch = []
        while frontier and len(batch) < a.round:
            reach, _, seq = heapq.heappop(frontier)
            k = canon(game(seq))
            if k in seen_opp:
                continue
            seen_opp.add(k)
            batch.append((seq, -reach))
        priors = gpu.run("prior", [s for s, _ in batch])
        children = []
        for (seq, reach), prior in zip(batch, priors):
            live = [i for i in prior if not game(seq + [i]).is_terminated()]
            bits += len(live)  # one "covered" digit per non-terminal reply
            cov = [i for i in live if prior[i] >= a.cover]
            if not cov:
                continue
            mass = sum(prior[i] for i in cov)
            for i in cov:
                q = 0.5 * prior[i] / mass + 0.5 / len(cov)
                children.append((seq + [i], reach * q))
        add_ours(children)
        rounds += 1
        el = time.time() - t0
        frac = min(1.0, bits / budget)
        eta = time.strftime("%H:%M", time.localtime(t0 + el / max(frac, 1e-6))) if frac > 0.02 else "?"
        depth = max(len(s) for s, _ in batch) if batch else 0
        say(f"round {rounds}: {sum(v is not None for v in ours.values())} of our positions, "
            f"{len(seen_opp)} opponent positions, {bits / 14:,.0f} of {a.chars:,} chars ({100 * frac:.0f}%), "
            f"reach down to {batch[-1][1] if batch else 0:.2e}, max ply {depth}, crossfish vetoes {vetoes}, ETA {eta}")

    gpu.close()
    with open(a.out, "w") as f:
        for seq, mv in (v for v in ours.values() if v is not None):
            f.write(f"S {','.join(map(str, seq))} {mv}\n")
    say(f"done: {len(ours)} of our positions, {len(seen_opp)} opponent positions, ~{bits / 14:,.0f} chars, "
        f"{vetoes} crossfish vetoes -> {a.out}")


if __name__ == "__main__":
    main()
