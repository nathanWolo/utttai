"""uttt.ai with the solver vs without, same network, fixed simulations per move.

Colour-swapped pairs from crossfish's SPRT book, CodinGame rules, CPU
onnxruntime. With a fixed simulation count the result does not depend on
timing, so games run in parallel.

usage: python solver_ab.py [pairs=200] [sims=100] [workers=8]
env: UTTTAI_NET
"""
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def play(args):
    pair, opening, sims = args
    import utttai_onnx as U
    sess = U.make_session()
    out = []
    for solver_is_x in (True, False):
        u = U.UltimateTicTacToe()
        for i in opening:
            u.execute(U.Action(symbol=u.next_symbol, index=i), verify=False)
        trees = {True: U.NMCTS(u, sims, sess, solver=True), False: U.NMCTS(u, sims, sess, solver=False)}
        while not u.is_terminated():
            solver_to_move = (u.next_symbol == 1) == solver_is_x
            t = trees[solver_to_move]
            t.synchronize(u)
            t.run()
            idx = t.best_action().index
            u.execute(U.Action(symbol=u.next_symbol, index=idx), verify=False)
        if u.is_result_draw():
            out.append(0.5)
        else:
            out.append(1.0 if (u.result == 1) == solver_is_x else 0.0)
    return out


def main():
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    sys.path.insert(0, str(HERE.parent))
    from bench_vs_utttai import load_openings
    openings = load_openings(pairs)
    scores = []
    with mp.Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(play, [(i, op, sims) for i, op in enumerate(openings)]), 1):
            scores.append(sum(res) / 2)
            if k % 25 == 0 or k == pairs:
                n = len(scores)
                mean = sum(scores) / n
                var = sum((x - mean) ** 2 for x in scores) / max(1, n - 1)
                se = math.sqrt(var / n)
                elo = lambda p: -400 * math.log10(1 / min(0.999, max(0.001, p)) - 1)
                print(f"{n} pairs: solver scores {mean:.3f}  Elo {elo(mean):+.1f} "
                      f"[{elo(mean - 1.96 * se):+.1f}, {elo(mean + 1.96 * se):+.1f}]", flush=True)


if __name__ == "__main__":
    main()
