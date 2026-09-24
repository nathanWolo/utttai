"""Does the MCTS-solver convert won positions the plain search let slip?

From a recorded match, take every game where crossfish announced a forced loss
(score <= -90000) yet did not lose: uttt.ai had a win and missed it. For the
position after crossfish's move (uttt.ai to move, and winning), run uttt.ai's
search for <ms> with the solver off and on, and ask a fresh crossfish search
(1 s, debug build) whether the chosen move keeps the win: crossfish to move
must still be proven lost. Positions crossfish itself does not re-prove lost
are skipped.

usage: python solver_check.py <match.live> <crossfish_debug.exe> [ms=90]
env: UTTTAI_NET, PATH (toolchain DLLs)
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import utttai_onnx as U  # noqa: E402

MATE = 90000


def cf_score(exe, moves, ms=1000):
    script = "\n".join(["NEW"] + [f"APPLY {i // 9} {i % 9}" for i in moves] + [f"THINK {ms}"]) + "\n"
    out = [t for t in subprocess.run([exe, "match"], input=script, capture_output=True, text=True).stdout.split()
           if t != "READY"]
    return int(out[3])


def search(sess, moves, ms, solver):
    u = U.UltimateTicTacToe()
    for i in moves:
        u.execute(U.Action(symbol=u.next_symbol, index=i), verify=False)
    s = U.NMCTS(u, 10 ** 9, sess, solver=solver)
    t0 = time.perf_counter()
    while (s.root.visit_count < 2 or time.perf_counter() - t0 < ms / 1000) and not s.solved():
        s._simulate()
    return s.best_action().index, s.root.proven, s.root.visit_count


def main():
    live, exe = sys.argv[1], sys.argv[2]
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    sess = U.make_session()
    cases = []
    for f in sorted(glob.glob(os.path.join(live, "g*.json"))):
        g = json.load(open(f))
        if g.get("status") != "done" or g["result"]["cg"] == -1:
            continue
        n0, mv = len(g["opening"]), g["moves"]
        k = next((j for j, m in enumerate(mv) if m["by"] == "cf" and m.get("e", 0) <= -MATE), None)
        if k is None:
            continue
        moves = g["opening"] + [m["i"] for m in mv[:k + 1]]  # through crossfish's move: uttt.ai to move
        if cf_score(exe, moves) < MATE:  # uttt.ai to move: crossfish must prove it winning for uttt.ai
            continue
        cases.append((g["id"], moves))
    print(f"{len(cases)} positions where uttt.ai was proven winning and let it slip in the game", flush=True)
    tally = {False: [0, 0], True: [0, 0]}
    for gid, moves in cases:
        row = [f"game {gid + 1:4d}"]
        for solver in (False, True):
            idx, proven, sims = search(sess, moves, ms, solver)
            keeps = cf_score(exe, moves + [idx]) <= -MATE  # crossfish to move, still proven lost
            tally[solver][0] += keeps
            tally[solver][1] += proven == 1.0
            row.append(f"{'solver' if solver else 'plain '}: {'keeps win' if keeps else 'LETS IT GO'}"
                       f"{' (root proven)' if proven == 1.0 else ''} [{sims} sims]")
        print("  " + "  |  ".join(row), flush=True)
    n = len(cases)
    for solver in (False, True):
        print(f"{'solver' if solver else 'plain '}: keeps the win in {tally[solver][0]}/{n}; "
              f"root proven won in {tally[solver][1]}/{n}")


if __name__ == "__main__":
    main()
