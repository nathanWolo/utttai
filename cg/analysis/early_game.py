"""How do crossfish and uttt.ai play the early game, and which is closer to right?

Positions: a seeded sample from crossfish's SPRT opening book (4-10 plies,
balanced). For each position:

  A. the two moves under test, one search at a time (timing is fair):
     crossfish 90 ms and uttt.ai 90 ms, fresh engines, no book; each repeated
     once to measure its own run-to-run noise
  B. references: crossfish 1 s and 5 s; uttt.ai 1,000 and 5,000 simulations
     (fixed counts, so these run in parallel without bias)
  C. where the two 90 ms moves differ, both moves are judged by a deep search
     of the resulting position from each engine: crossfish 2 s (score) and
     uttt.ai 3,000 simulations (value)

usage: python early_game.py <crossfish_debug.exe> <out.json> [positions=300] [workers=8]
env: UTTTAI_NET, PATH (toolchain DLLs)
"""
import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
BOOK = Path(r"C:\Users\natha\crossfish\cpp_impl\opening_book.bin")
MATE = 90000
CF = None  # set in main / worker init


def book_positions(n, seed=20260924):
    data = BOOK.read_bytes()
    count = int.from_bytes(data[8:12], "little")
    idx = random.Random(seed).sample(range(count), n)
    return [list(data[32 + 16 * k + 1: 32 + 16 * k + 1 + data[32 + 16 * k]]) for k in idx]


def cf_think(moves, ms):
    script = "\n".join(["NEW"] + [f"APPLY {i // 9} {i % 9}" for i in moves] + [f"THINK {ms}"]) + "\n"
    out = [t for t in subprocess.run([CF, "match"], input=script, capture_output=True, text=True).stdout.split()
           if t != "READY"]
    return int(out[0]) * 9 + int(out[1]), int(out[3]), int(out[2])  # move, score (mover's view), depth


def ut_search(moves, ms=None, sims=None):
    import utttai_onnx as U
    global SESS
    u = U.UltimateTicTacToe()
    for i in moves:
        u.execute(U.Action(symbol=u.next_symbol, index=i), verify=False)
    s = U.NMCTS(u, sims or 10 ** 9, SESS)
    if ms:
        t0 = time.perf_counter()
        while s.root.visit_count < 2 or time.perf_counter() - t0 < ms / 1000:
            s._simulate()
    else:
        s.run()
    # root value is for the side to move (the mover)
    return s.best_action().index, s.root.state_value_mean, s.root.visit_count


def init_worker(cf):
    global CF, SESS
    CF = cf
    import utttai_onnx as U
    SESS = U.make_session()


def references(moves):
    return dict(cf1s=cf_think(moves, 1000)[0], cf5s=cf_think(moves, 5000)[0],
                ut1k=ut_search(moves, sims=1000)[0], ut5k=ut_search(moves, sims=5000)[0])


def judge(args):
    """Both engines' deep verdict on each candidate move, from the mover's side."""
    moves, cand = args
    out = {}
    for name, mv in cand.items():
        after = moves + [mv]
        _, cf_s, _ = cf_think(after, 2000)       # opponent to move: negate
        _, ut_v, _ = ut_search(after, sims=3000)
        out[name] = dict(cf=-cf_s, ut=-ut_v)
    return out


def main():
    cf, out_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    init_worker(cf)
    pos = book_positions(n)
    rows = [dict(moves=m, ply=len(m)) for m in pos]

    t0 = time.time()
    for k, r in enumerate(rows):  # A: timed moves, strictly one search at a time
        r["cf90"], r["cf90_score"], r["cf90_depth"] = cf_think(r["moves"], 90)
        r["ut90"], r["ut90_value"], r["ut90_sims"] = ut_search(r["moves"], ms=90)
        r["cf90b"] = cf_think(r["moves"], 90)[0]
        r["ut90b"] = ut_search(r["moves"], ms=90)[0]
        if (k + 1) % 50 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] A {k + 1}/{n} positions ({time.time() - t0:.0f} s)", flush=True)

    with mp.Pool(workers, initializer=init_worker, initargs=(cf,)) as pool:  # B: references
        for k, ref in enumerate(pool.imap(references, [r["moves"] for r in rows])):
            rows[k].update(ref)
            if (k + 1) % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] B {k + 1}/{n} positions", flush=True)
        dis = [k for k, r in enumerate(rows) if r["cf90"] != r["ut90"]]
        jobs = [(rows[k]["moves"], {"cf90": rows[k]["cf90"], "ut90": rows[k]["ut90"]}) for k in dis]
        for j, (k, res) in enumerate(zip(dis, pool.imap(judge, jobs))):  # C: judge disagreements
            rows[k]["judge"] = res
            if (j + 1) % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] C {j + 1}/{len(dis)} disagreements", flush=True)

    json.dump(rows, open(out_path, "w"))
    report(rows)


def report(rows):
    n = len(rows)
    pct = lambda x: f"{100 * x / n:5.1f}%"
    agree = lambda a, b: sum(r[a] == r[b] for r in rows)
    print(f"\n{n} opening-book positions (plies {min(r['ply'] for r in rows)}-{max(r['ply'] for r in rows)})")
    print(f"self-agreement at 90 ms (run-to-run noise): crossfish {pct(agree('cf90', 'cf90b'))}, "
          f"uttt.ai {pct(agree('ut90', 'ut90b'))}")
    print(f"crossfish 90 ms and uttt.ai 90 ms choose the same move: {pct(agree('cf90', 'ut90'))}")
    print("agreement with each reference:        crossfish 90 ms   uttt.ai 90 ms")
    for ref, label in (("cf1s", "crossfish 1 s"), ("cf5s", "crossfish 5 s"), ("ut1k", "uttt.ai 1,000 sims"),
                       ("ut5k", "uttt.ai 5,000 sims")):
        print(f"  {label:34s} {pct(agree('cf90', ref)):>10s}   {pct(agree('ut90', ref)):>12s}")
    print(f"references agree with each other: cf5s=ut5k {pct(agree('cf5s', 'ut5k'))}, "
          f"cf1s=cf5s {pct(agree('cf1s', 'cf5s'))}, ut1k=ut5k {pct(agree('ut1k', 'ut5k'))}")
    both = [r for r in rows if r["cf5s"] == r["ut5k"]]
    if both:
        print(f"where both strong references agree ({len(both)} positions): crossfish 90 ms matches "
              f"{100 * sum(r['cf90'] == r['cf5s'] for r in both) / len(both):.1f}%, uttt.ai 90 ms "
              f"{100 * sum(r['ut90'] == r['cf5s'] for r in both) / len(both):.1f}%")
    judged = [r for r in rows if "judge" in r]
    if judged:
        def prefers(r, j):
            a, b = r["judge"]["cf90"][j], r["judge"]["ut90"][j]
            return "cf90" if a > b else "ut90" if b > a else "tie"
        for j, label in (("cf", "crossfish 2 s"), ("ut", "uttt.ai 3,000 sims")):
            c = Counter(prefers(r, j) for r in judged)
            print(f"when the 90 ms moves differ ({len(judged)}), {label} prefers crossfish's move "
                  f"{c['cf90']}, uttt.ai's {c['ut90']}, tie {c['tie']}")
        c = Counter((prefers(r, "cf"), prefers(r, "ut")) for r in judged)
        print(f"both judges agree: crossfish's move better {c[('cf90', 'cf90')]}, uttt.ai's better "
              f"{c[('ut90', 'ut90')]}; judges split {len(judged) - c[('cf90', 'cf90')] - c[('ut90', 'ut90')]}")
    by_ply = {}
    for r in rows:
        by_ply.setdefault(r["ply"], []).append(r)
    print("by ply (agreement with crossfish 5 s / uttt.ai 5,000 sims):")
    for p in sorted(by_ply):
        g = by_ply[p]
        f = lambda a, b: 100 * sum(x[a] == x[b] for x in g) / len(g)
        print(f"  ply {p:2d} ({len(g):3d}): cf90 {f('cf90', 'cf5s'):5.1f}% / {f('cf90', 'ut5k'):5.1f}%   "
              f"ut90 {f('ut90', 'cf5s'):5.1f}% / {f('ut90', 'ut5k'):5.1f}%")
    style = lambda key: Counter("centre square" if r[key] % 9 == 4 else "corner square" if r[key] % 9 in (0, 2, 6, 8)
                                else "edge square" for r in rows)
    sends = lambda key: sum(r[key] % 9 == 4 for r in rows)
    print(f"style: crossfish {dict(style('cf90'))}, sends to centre board {sends('cf90')}; "
          f"uttt.ai {dict(style('ut90'))}, sends to centre board {sends('ut90')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.load(open(sys.argv[2])))
    else:
        main()
