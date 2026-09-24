"""Promotion gate: SPRT of network A vs network B on colour-swapped opening pairs.

Openings stream from crossfish's 50,000-position SPRT book in a fixed shuffled
order; each opening is played twice with colours swapped (same search as
netmatch.py: uttt.ai's NMCTS, best-move play, no noise, CodinGame rules).
The statistic is pentanomial: each pair scores 0, 1/4, 1/2, 3/4 or 1 for A, and
LLR = N (s1 - s0) (2 mean - s0 - s1) / (2 var) over completed pairs, with
s = 1 / (1 + 10^(-elo / 400)) (the usual normal approximation to the GSPRT). Mean and
variance include a PRIOR pseudo-count per outcome, and no decision before MIN_PAIRS.
Stops when LLR reaches +bound (accept H1: A is stronger) or -bound (accept H0).
Pairs are counted strictly in opening order: pair i enters the statistic only once
every pair before it has finished. Counting pairs as they complete would favour
short games (mostly decisive ones) and bias an early stop.

usage: sprt_gate.py <netA> <netB> <sims> <workers> <log> [--elo0 0] [--elo1 10] [--bound 3]
                    [--max-pairs 5000] [--games-per-worker 128] [--backend ort|torch]
last log line: "SPRT H1 accepted" / "SPRT H0 accepted" / "SPRT inconclusive"
"""
import math
import random
import sys
import time

import netmatch
from netmatch import Match, BOOK, elo


def all_openings():
    data = BOOK.read_bytes()
    count = int.from_bytes(data[8:12], "little")
    order = list(range(count))
    random.Random(20260923).shuffle(order)
    return [list(data[32 + 16 * k + 1: 32 + 16 * k + 1 + data[32 + 16 * k]]) for k in order]


def worker(wid, jobs, results, net_a, net_b, sims, batch, backend):
    # netmatch.worker with the evaluator backend selectable
    orig = netmatch.Evaluator
    netmatch.Evaluator = lambda path: orig(path, backend)
    netmatch.worker(wid, jobs, results, net_a, net_b, sims, batch)


PRIOR = 0.5  # pseudo-count on each of the five pair outcomes: keeps early variance from collapsing to 0
MIN_PAIRS = 50


def llr(pairs, elo0, elo1):
    n = len(pairs)
    if n < 2:
        return 0.0
    w = [PRIOR] * 5
    for x in pairs:
        w[round(x * 4)] += 1
    tot = sum(w)
    mean = sum(k / 4 * c for k, c in enumerate(w)) / tot
    var = sum((k / 4 - mean) ** 2 * c for k, c in enumerate(w)) / tot
    s0, s1 = (1 / (1 + 10 ** (-e / 400)) for e in (elo0, elo1))
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def main():
    import multiprocessing as mp
    args, opts = [], dict(elo0=0.0, elo1=10.0, bound=3.0, max_pairs=5000, games_per_worker=128, backend="ort")
    it = iter(sys.argv[1:])
    for a in it:
        if a.startswith("--"):
            k = a[2:].replace("-", "_")
            opts[k] = type(opts[k])(next(it))
        else:
            args.append(a)
    net_a, net_b, sims, workers, log_path = args[0], args[1], int(args[2]), int(args[3]), args[4]

    openings = all_openings()[:opts["max_pairs"]]
    jobs, results = mp.Queue(), mp.Queue()
    for i, op in enumerate(openings):  # a pair's two games are adjacent in the queue
        jobs.put((2 * i, op, True))
        jobs.put((2 * i + 1, op, False))
    for _ in range(workers):
        jobs.put(None)
    procs = [mp.Process(target=worker, args=(w, jobs, results, net_a, net_b, sims, opts["games_per_worker"],
                                             opts["backend"])) for w in range(workers)]
    for p in procs:
        p.start()
    log = open(log_path, "w", encoding="utf-8")

    def out(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    out(f"SPRT A={net_a} vs B={net_b}; {sims} simulations; H0 elo {opts['elo0']:g}, H1 elo {opts['elo1']:g}, "
        f"LLR bounds +/-{opts['bound']:g}; pairs of colour-swapped games (CodinGame rules), max {len(openings)} pairs")
    t0 = time.time()
    done, pairs, games = {}, [], []  # done: game id -> result; games: results of counted pairs
    verdict = "inconclusive"
    last_report = 0
    while len(pairs) < len(openings):
        gid, a_is_x, r = results.get()
        done[gid] = r
        grew = False
        while 2 * len(pairs) in done and 2 * len(pairs) + 1 in done:
            r1, r2 = done.pop(2 * len(pairs)), done.pop(2 * len(pairs) + 1)
            games += [r1, r2]
            pairs.append(((r1 + 1) / 2 + (r2 + 1) / 2) / 2)
            grew = True
        if not grew:
            continue
        value = llr(pairs, opts["elo0"], opts["elo1"])
        decided = len(pairs) >= MIN_PAIRS and abs(value) >= opts["bound"]
        if decided or len(pairs) - last_report >= 10:
            last_report = len(pairs)
            w, d, l = games.count(1), games.count(0), games.count(-1)
            e, ci = elo(w, d, l)
            penta = [sum(1 for x in pairs if abs(x - v) < 1e-9) for v in (0, 0.25, 0.5, 0.75, 1)]
            rate = (len(games) + len(done)) / max(1e-9, time.time() - t0) * 3600
            out(f"[{time.strftime('%H:%M:%S')}] {len(pairs)} pairs ({len(games) + len(done)} games done, "
                f"{len(done)} waiting on earlier pairs, {rate:,.0f}/h)  "
                f"A W/D/L {w}/{d}/{l}  penta {penta}  Elo(A) {e:+.1f} +/- {ci:.1f}  "
                f"LLR {value:+.2f} [{-opts['bound']:g}, {opts['bound']:g}]")
        if decided:
            verdict = "H1 accepted" if value > 0 else "H0 accepted"
            break
    out(f"SPRT {verdict}")
    for p in procs:
        p.terminate()
    jobs.cancel_join_thread()  # unplayed openings are still queued; do not block exit flushing them
    results.cancel_join_thread()
    log.close()


if __name__ == "__main__":
    main()
