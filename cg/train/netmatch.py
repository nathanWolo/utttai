"""Network A vs network B: uttt.ai's search (CodinGame-rules fork) with each
network, best-move play (no noise, no temperature), tree reuse per side, from
crossfish's balanced SPRT openings with colors swapped. Leaves from all games
are evaluated in two GPU batches per round (one per network).

usage: netmatch.py <netA> <netB> <openings> <sims> <workers> <log> [skip]

skip: extend an earlier match that used <skip> openings: pick <openings> new ones
      disjoint from that run's, so the two results can be pooled
"""
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

import selfplay
from selfplay import Evaluator, Game, encode
from utttpy.game.action import Action
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe

BOOK = Path(r"C:\Users\natha\crossfish\cpp_impl\opening_book.bin")


def load_openings(n, skip=0):
    data = BOOK.read_bytes()
    count = int.from_bytes(data[8:12], "little")
    if skip:
        used = {i * (count // skip) for i in range(skip)}
        step = count // (n + skip)
        idx = [j * step for j in range(n + skip) if j * step not in used][:n]
        assert len(idx) == n, "not enough disjoint openings"
    else:
        idx = [i * (count // n) for i in range(n)]
    return [list(data[32 + 16 * k + 1: 32 + 16 * k + 1 + data[32 + 16 * k]]) for k in idx]


class Side(Game):
    """One player's search tree; reuses Game's select/expand/backup without noise."""

    def __init__(self, u, sims, rng):
        super().__init__(rng, sims)
        self.u = u.clone()
        self.root = selfplay.Node(u.clone())

    def _noisy(self, p):
        return p

    def advance(self, idx, u):
        if self.root.children:
            for c in self.root.children:
                if c.action == idx:
                    if c.u is None:
                        c.u = u.clone()
                    self.root = c
                    return
        self.root = selfplay.Node(u.clone())

    def best(self):
        top = max(c.n for c in self.root.children)
        return self.rng.choice([c for c in self.root.children if c.n >= top]).action


class Match:
    def __init__(self, gid, opening, a_is_x, sims):
        self.gid, self.a_is_x = gid, a_is_x
        rng = random.Random(gid)
        self.u = UltimateTicTacToe()
        for idx in opening:
            self.u.execute(Action(symbol=self.u.next_symbol, index=idx))
        self.sides = {"A": Side(self.u, sims, rng), "B": Side(self.u, sims, rng)}
        self.done = False

    def to_move(self):
        return "A" if (self.u.next_symbol == 1) == self.a_is_x else "B"

    def result(self):
        """+1 A wins, 0 draw, -1 B wins (CodinGame rules via the fork)."""
        if self.u.is_result_draw():
            return 0
        a_sym = 1 if self.a_is_x else 2
        return 1 if self.u.result == a_sym else -1


def worker(wid, jobs, results, net_a, net_b, sims, batch):
    import warnings
    warnings.filterwarnings("ignore")
    evs = {"A": Evaluator(net_a), "B": Evaluator(net_b)}
    live = []
    no_more_jobs = False
    while True:
        while len(live) < batch and not no_more_jobs:
            job = jobs.get()
            if job is None:
                no_more_jobs = True
                break
            live.append(Match(*job, sims))
        if not live:
            break
        pend = {"A": [], "B": []}
        for m in live:
            side = m.sides[m.to_move()]
            for _ in range(4):
                leaf = side.select()
                if leaf is not None:
                    pend[m.to_move()].append((side, encode(leaf.u, leaf.legal)))
                    break
        for k in ("A", "B"):
            if pend[k]:
                logits, values = evs[k]([x for _, x in pend[k]])
                for (side, _), lg, v in zip(pend[k], logits, values):
                    side.expand_and_backup(lg, v)
        nxt = []
        for m in live:
            side = m.sides[m.to_move()]
            if side.pending is None and side.search_done():
                idx = side.best()
                m.u.execute(Action(symbol=m.u.next_symbol, index=idx), verify=False)
                for s in m.sides.values():
                    s.advance(idx, m.u)
                if m.u.is_terminated():
                    results.put((m.gid, m.a_is_x, m.result()))
                    continue
            nxt.append(m)
        live = nxt


def elo(w, d, l):
    n = w + d + l
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s * s) / n
    sd = math.sqrt(var / n)
    f = lambda p: -400 * math.log10(1 / min(0.999, max(0.001, p)) - 1)
    return f(s), (f(s + 1.96 * sd) - f(s - 1.96 * sd)) / 2


def main():
    import multiprocessing as mp
    net_a, net_b, n_open, sims, workers, log_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    openings = load_openings(n_open, int(sys.argv[7]) if len(sys.argv) > 7 else 0)
    jobs, results = mp.Queue(), mp.Queue()
    total = 2 * n_open
    for i, op in enumerate(openings):
        jobs.put((2 * i, op, True))
        jobs.put((2 * i + 1, op, False))
    for _ in range(workers):
        jobs.put(None)
    per_worker_batch = max(1, total // workers + 1)
    procs = [mp.Process(target=worker, args=(w, jobs, results, net_a, net_b, sims, per_worker_batch)) for w in range(workers)]
    for p in procs:
        p.start()
    log = open(log_path, "w", encoding="utf-8")

    def out(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    out(f"A={net_a} vs B={net_b}; {sims} simulations each; {total} games (CodinGame rules)")
    t0 = time.time()
    rows = []
    for k in range(1, total + 1):
        rows.append(results.get())
        if k % 20 == 0 or k == total:
            r = [x[2] for x in rows]
            w, d, l = r.count(1), r.count(0), r.count(-1)
            e, ci = elo(w, d, l)
            eta = time.strftime("%H:%M", time.localtime(t0 + (time.time() - t0) / k * total))
            out(f"[{time.strftime('%H:%M:%S')}] {k}/{total} games  ETA {eta}  A W/D/L {w}/{d}/{l}  Elo(A) {e:+.1f} +/- {ci:.1f}")
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
