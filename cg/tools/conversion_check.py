"""Can uttt.ai convert won positions against crossfish?

Starts from the positions solver_check.py uses (uttt.ai to move, proven
winning, a win it let slip in the recorded game) and plays each to the end:
uttt.ai with the solver off and then on, crossfish (the metadata build) at the
same time per move, one game at a time. Reports how often uttt.ai wins, draws
or loses from a won position.

usage: python conversion_check.py <match.live> <crossfish_debug.exe> <crossfish_meta.exe> [ms=90]
env: UTTTAI_NET, PATH (toolchain DLLs)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import utttai_onnx as U  # noqa: E402
from solver_check import MATE, cf_score  # noqa: E402

import glob, json, os  # noqa: E402,E401


def won_positions(live, exe):
    out = []
    for f in sorted(glob.glob(os.path.join(live, "g*.json"))):
        g = json.load(open(f))
        if g.get("status") != "done" or g["result"]["cg"] == -1:
            continue
        mv = g["moves"]
        k = next((j for j, m in enumerate(mv) if m["by"] == "cf" and m.get("e", 0) <= -MATE), None)
        if k is None:
            continue
        moves = g["opening"] + [m["i"] for m in mv[:k + 1]]
        if cf_score(exe, moves) >= MATE:
            out.append((g["id"], moves))
    return out


def play_out(sess, meta_exe, moves, ms, solver):
    """uttt.ai to move from `moves`; returns +1 uttt.ai wins, 0 draw, -1 loss."""
    u = U.UltimateTicTacToe()
    for i in moves:
        u.execute(U.Action(symbol=u.next_symbol, index=i), verify=False)
    ut_sym = u.next_symbol
    cf = subprocess.Popen([meta_exe, "match"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
    cf.stdin.write("NEW\n" + "".join(f"APPLY {i // 9} {i % 9}\n" for i in moves))
    cf.stdin.flush()
    cf.stdout.readline()  # READY
    search = U.NMCTS(u, 10 ** 9, sess, solver=solver)
    try:
        while not u.is_terminated():
            if u.next_symbol == ut_sym:
                search.synchronize(u)
                before, t0 = search.root.visit_count, time.perf_counter()
                while (search.root.visit_count - before < 2 or time.perf_counter() - t0 < ms / 1000) \
                        and not search.solved():
                    search._simulate()
                idx = search.best_action().index
                cf.stdin.write(f"APPLY {idx // 9} {idx % 9}\n")
                cf.stdin.flush()
            else:
                cf.stdin.write(f"GO {ms}\n")
                cf.stdin.flush()
                r = cf.stdout.readline().split()
                idx = int(r[0]) * 9 + int(r[1])
            u.execute(U.Action(symbol=u.next_symbol, index=idx), verify=False)
    finally:
        cf.kill()
    if u.is_result_draw():
        return 0
    return 1 if u.result == ut_sym else -1


def main():
    live, dbg, meta = sys.argv[1], sys.argv[2], sys.argv[3]
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 90
    sess = U.make_session()
    cases = won_positions(live, dbg)
    print(f"{len(cases)} proven-won positions uttt.ai let slip in the recorded match", flush=True)
    for solver in (False, True):
        res = [play_out(sess, meta, moves, ms, solver) for _, moves in cases]
        w, d, l = res.count(1), res.count(0), res.count(-1)
        print(f"{'solver' if solver else 'plain '}: from {len(cases)} won positions uttt.ai won {w}, drew {d}, lost {l}",
              flush=True)


if __name__ == "__main__":
    main()
