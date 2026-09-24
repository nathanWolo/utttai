"""Hill-climb the CodinGame-rules uttt.ai network until returns diminish.

Per generation g:
  1. self-play with the current best network (GAMES games, SIMS simulations)
  2. train a candidate from the best on the last WINDOW generations of data,
     anchored to the best's policy (ANCHOR_WEIGHT)
  3. gate: SPRT of candidate vs best on colour-swapped opening pairs at 1,000
     simulations (sprt_gate.py: H0 0 Elo, H1 +10 Elo, LLR bounds +/-3); promote on H1
Self-play already completed for a generation (its log shows all games) is reused.
Stops after two consecutive generations that are not promoted.
Benchmarks the best network against crossfish (1 s per move each, one game at a time)
after the second accepted generation and at the end.

usage: python loop.py <first_gen> <best.pt> <max_gen>
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_GPU = r"C:\Users\natha\crossfish\toolchains\py312-dml\Scripts\python.exe"
PY_CPU = sys.executable  # has onnxruntime for the crossfish benchmark
TOOLCHAIN = r"C:\Users\natha\crossfish\toolchains\llvm-mingw-20260616-ucrt-x86_64\bin"
GAMES, SIMS, WORKERS = 5000, 800, 6
# Self-play: onnxruntime DirectML, 8 workers x 256 games (bench: 42.5k evals/s vs 28.7k torch 6x128).
SP_WORKERS, SP_BATCH, SP_FLAGS = 8, 256, ["--backend", "ort"]
WINDOW, EPOCHS, LR, ANCHOR_WEIGHT = 3, 2, 1e-4, 0.25
GATE_SIMS, GATE_WORKERS = 1000, 6
LOG = HERE / "loop.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, logfile, env=None, cwd=HERE):
    log(f"  $ {' '.join(str(c) for c in cmd)}  (log: {logfile})")
    with open(HERE / logfile, "w", encoding="utf-8") as out:
        rc = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=out, stderr=subprocess.DEVNULL,
                            env=env).returncode
    if rc != 0:
        log(f"  FAILED with exit code {rc}; stopping")
        sys.exit(1)


def last_line(logfile, pattern):
    lines = [l for l in (HERE / logfile).read_text(encoding="utf-8").splitlines() if re.search(pattern, l)]
    return lines[-1] if lines else ""


def crossfish_benchmark(net_onnx, tag):
    env = dict(os.environ, UTTTAI_NET=f"train/{net_onnx}",
               PATH=TOOLCHAIN + os.pathsep + os.environ["PATH"])
    logname = f"cf_{tag}.log"
    # One game at a time: parallel games slow uttt.ai far more than crossfish (see speed_probe.py).
    log(f"crossfish benchmark ({tag}): {net_onnx}, 1 s per move each, 100 games, one at a time, CodinGame rules")
    run([PY_CPU, "bench_vs_utttai.py", "1000ms", 1000, 50, 1, f"train/{logname}", "crossfish_cg_meta.exe"],
        f"cf_{tag}.stdout", env=env, cwd=HERE.parent)
    log("  " + last_line(logname, r"^CodinGame rules").strip())


def main():
    first, best, max_gen = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    log(f"=== hill-climb start: best {best}, generations {first}..{max_gen}, "
        f"{GAMES} games x {SIMS} sims, window {WINDOW}, anchor {ANCHOR_WEIGHT}")
    weak_streak, accepted = 0, 0
    for g in range(first, max_gen + 1):
        t0 = time.time()
        if (HERE / f"gen{g}.log").exists() and f"{GAMES}/{GAMES} games" in (HERE / f"gen{g}.log").read_text(encoding="utf-8"):
            log(f"--- generation {g}: reusing completed self-play in data/gen{g}")
        else:
            log(f"--- generation {g}: self-play with {best}")
            run([PY_GPU, "selfplay.py", best, f"data/gen{g}", GAMES, SIMS, SP_WORKERS, SP_BATCH, *SP_FLAGS], f"gen{g}.log")
        dirs = ",".join(f"data/gen{k}" for k in range(max(1, g - WINDOW + 1), g + 1) if (HERE / f"data/gen{k}").exists())
        log(f"  training net{g} on {dirs}")
        run([PY_GPU, "train.py", best, dirs, f"net{g}", EPOCHS, LR, best, ANCHOR_WEIGHT], f"train{g}.log")
        log("  " + last_line(f"train{g}.log", r"validation after epoch").split("] ", 1)[-1])
        run([PY_GPU, "sprt_gate.py", f"net{g}.pt", best, GATE_SIMS, GATE_WORKERS, f"gate{g}.log"], f"gate{g}.stdout")
        status = last_line(f"gate{g}.log", r"pairs \(").split("]  ", 1)[-1]
        promoted = last_line(f"gate{g}.log", r"^SPRT (H0|H1|inconclusive)").strip() == "SPRT H1 accepted"
        mins = (time.time() - t0) / 60
        if promoted:
            best = f"net{g}.pt"
            accepted += 1
            weak_streak = 0
            log(f"  PROMOTED net{g} (SPRT H1): {status}  ({mins:.0f} min)")
        else:
            weak_streak += 1
            log(f"  not promoted net{g}: {status}; keeping {best}  ({mins:.0f} min)")
        if accepted == 2 and promoted:
            crossfish_benchmark(best.replace(".pt", ".onnx"), f"mid_{best[:-3]}")
        if weak_streak >= 2:
            log("diminishing returns: two consecutive generations not promoted")
            break
    crossfish_benchmark(best.replace(".pt", ".onnx"), f"final_{best[:-3]}")
    log(f"=== hill-climb done: best network {best}")


if __name__ == "__main__":
    main()
