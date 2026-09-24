"""Compare two crossfish debug builds on positions from a recorded match.

For each sampled crossfish turn, a fresh engine searches for <ms> with CF_TRACE
on; the probe counts aspiration fail-lows whose window lies in mate range
(alpha >= 90000), whether the final reported score is an aspiration bound
(20000 <= |score| < 90000, i.e. time ran out inside a fail-low cascade), and
the depth reached. Positions come in two groups: crossfish turns after its
first mate-range score in the game ("mate phase") and ordinary middlegame
turns (plies 12-30) before any mate score.

usage: python mate_window_probe.py <match.live> <exe_before> <exe_after> [n_per_group=150] [ms=90]
env: PATH must reach the crossfish toolchain DLLs.
"""
import glob
import json
import os
import random
import re
import statistics
import subprocess
import sys

MATE = 90000


def positions(live, n, seed=7):
    mate, mid = [], []
    for f in sorted(glob.glob(os.path.join(live, "g*.json"))):
        g = json.load(open(f))
        if g.get("status") != "done":
            continue
        n0 = len(g["opening"])
        seq = g["opening"] + [m["i"] for m in g["moves"]]
        seen_mate = False
        for j, m in enumerate(g["moves"]):
            if m["by"] != "cf" or m.get("book"):
                continue
            k = n0 + j
            if seen_mate:
                mate.append(seq[:k])
            elif 12 <= k <= 30:
                mid.append(seq[:k])
            if abs(m.get("e", 0)) >= MATE:
                seen_mate = True
    rng = random.Random(seed)
    return rng.sample(mate, min(n, len(mate))), rng.sample(mid, min(n, len(mid)))


def probe(exe, moves, ms):
    script = "\n".join(["NEW"] + [f"APPLY {i // 9} {i % 9}" for i in moves] + [f"THINK {ms}"]) + "\n"
    p = subprocess.run([exe, "match"], input=script, capture_output=True, text=True,
                       env=dict(os.environ, CF_TRACE="1"))
    out = [l for l in p.stdout.split() if l != "READY"]
    move, depth, score = int(out[0]) * 9 + int(out[1]), int(out[2]), int(out[3])
    mate_fail_lows = 0
    for line in p.stderr.splitlines():
        r = re.match(r"iter depth \d+ window \[(-?\d+), (-?\d+)\] -> -?\d+ FAIL-LOW", line)
        if r and int(r.group(1)) >= MATE:
            mate_fail_lows += 1
    return dict(move=move, depth=depth, score=score, mfl=mate_fail_lows, bound=20000 <= abs(score) < MATE)


def summary(rows):
    return (f"mate-window fail-lows/search {statistics.mean(r['mfl'] for r in rows):5.2f}  "
            f"searches with any {sum(r['mfl'] > 0 for r in rows):3d}  "
            f"bound reported as score {sum(r['bound'] for r in rows):3d}  "
            f"median depth {statistics.median(r['depth'] for r in rows):4.1f}")


def main():
    live, before, after = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    ms = int(sys.argv[5]) if len(sys.argv) > 5 else 90
    mate, mid = positions(live, n)
    for name, group in (("mate phase", mate), ("middlegame", mid)):
        res = {b: [probe(exe, pos, ms) for pos in group] for b, exe in (("before", before), ("after", after))}
        same = sum(a["move"] == b["move"] for a, b in zip(res["before"], res["after"]))
        print(f"{name} ({len(group)} positions, {ms} ms):")
        for b in ("before", "after"):
            print(f"  {b:6s} {summary(res[b])}")
        print(f"  same move chosen: {same}/{len(group)}", flush=True)


if __name__ == "__main__":
    main()
