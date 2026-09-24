#!/usr/bin/env bash
# End-to-end self-play benchmarks, one change at a time from the baseline.
# Each: 30 s warm-up + 120 s measured, net2, 800 simulations, idle machine.
cd "$(dirname "$0")"
PY=/c/Users/natha/crossfish/toolchains/py312-dml/Scripts/python.exe
run() {
  echo "[$(date +%H:%M:%S)] $*"
  "$PY" selfplay.py net2.pt none 0 800 "$@" --bench 120 2>/dev/null | grep -a -A2 "^BENCH"
}
run 6 128                    # B0  baseline
run 6 128                    # B0' baseline repeat (noise)
run 6 128 --backend ort      # B1  onnxruntime DirectML
run 6 256 --backend ort      # B1' onnxruntime, batch 256 (its best micro-benchmark size)
run 6 256                    # B4  torch, 256 games per worker
run 6 64                     # B4' torch, 64 games per worker
run 10 128                   # B3  torch, 10 workers
run 12 128                   # B3' torch, 12 workers
echo "[$(date +%H:%M:%S)] done"
