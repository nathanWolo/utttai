#!/usr/bin/env bash
# crossfish vs uttt.ai net4 at equal wall time per move (one core each, one game at a time: parallel games slow uttt.ai far more than crossfish), three time controls.
cd "$(dirname "$0")"
export UTTTAI_NET=train/net4.onnx PATH="/c/Users/natha/crossfish/toolchains/llvm-mingw-20260616-ucrt-x86_64/bin:$PATH"
LOG=train/bench_net4.log
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
run() {  # ms games
  ts "start: ${1} ms per move, $2 games (log train/bench_net4_${1}ms.log)"
  python3 bench_vs_utttai.py ${1}ms $1 $(( $2 / 2 )) 1 train/bench_net4_${1}ms.log crossfish_cg_meta.exe > train/bench_net4_${1}ms.stdout 2>&1
  grep -a "rules *:\|full board\|mean think" train/bench_net4_${1}ms.log | tr -d '\r' | sed 's/^/    /' | tee -a $LOG
}
ts "=== crossfish vs net4, equal time, one game at a time"
run 90 1000
run 1000 500
run 5000 100
ts "=== done"
