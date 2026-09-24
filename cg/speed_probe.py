"""How much do parallel games slow each engine? Runs K processes at once; each
alternates timed uttt.ai searches (sims/s) and crossfish searches (nodes/s) on book
openings, like a match worker does, and reports medians per K.

usage: python speed_probe.py <K,K,...> [seconds_per_search=1]
env: UTTTAI_NET, PATH (toolchain DLLs for the crossfish binary)
"""
import multiprocessing as mp
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = str(HERE / "crossfish_cg_meta.exe")


def probe(wid, secs, reps, out):
    sys.path.insert(0, str(HERE))
    from bench_vs_utttai import load_openings
    from utttai_onnx import NMCTS, make_session, UltimateTicTacToe
    from utttpy.game.action import Action
    sess = make_session()
    proc = subprocess.Popen([EXE, "match"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    ut, cf = [], []
    for k, op in enumerate(load_openings(64)[wid * reps:(wid + 1) * reps]):
        u = UltimateTicTacToe()
        proc.stdin.write("NEW\n"); proc.stdin.flush(); proc.stdout.readline()
        for idx in op:
            u.execute(Action(symbol=u.next_symbol, index=idx))
            proc.stdin.write(f"APPLY {idx // 9} {idx % 9}\n")
        search = NMCTS(u, 10 ** 9, sess)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < secs:
            search._simulate()
        ut.append(search.root.visit_count / (time.perf_counter() - t0))
        t0 = time.perf_counter()
        proc.stdin.write(f"GO {int(secs * 1000)}\n"); proc.stdin.flush()
        reply = proc.stdout.readline().split()
        cf.append(int(reply[4]) / (time.perf_counter() - t0))
    proc.kill()
    out.put((ut, cf))


def main():
    ks = [int(k) for k in sys.argv[1].split(",")]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    base = None
    for k in ks:
        out = mp.Queue()
        ps = [mp.Process(target=probe, args=(w, secs, 3, out)) for w in range(k)]
        for p in ps:
            p.start()
        res = [out.get() for _ in ps]
        for p in ps:
            p.join()
        ut = statistics.median(x for r in res for x in r[0])
        cf = statistics.median(x for r in res for x in r[1])
        base = base or (ut, cf)
        print(f"K={k}: uttt.ai {ut:,.0f} sims/s ({ut / base[0]:.0%})   crossfish {cf / 1e6:.2f}M nodes/s ({cf / base[1]:.0%})", flush=True)


if __name__ == "__main__":
    main()
