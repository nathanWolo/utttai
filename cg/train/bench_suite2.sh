#!/usr/bin/env bash
# Round 2: onnxruntime with more workers (Python-side select/backup is now ~50% of worker time).
cd "$(dirname "$0")"
PY=/c/Users/natha/crossfish/toolchains/py312-dml/Scripts/python.exe
run() {
  echo "[$(date +%H:%M:%S)] $*"
  "$PY" selfplay.py net2.pt none 0 800 "$@" --bench 120 2>/dev/null | grep -a -A2 "^BENCH"
}
run 6 256 --backend ort      # repeat of round-1 best (noise check)
run 10 128 --backend ort
run 8 256 --backend ort
run 12 128 --backend ort
echo "[$(date +%H:%M:%S)] done"
