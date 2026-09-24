# uttt.ai for CodinGame rules

This branch (`codingame-rules`) adapts uttt.ai to the rules of CodinGame's
Ultimate Tic-Tac-Toe and continues training its policy-value network under
them. It was built to measure uttt.ai against
[crossfish](https://github.com/nathanWolo/crossfish), a CodinGame
alpha-beta engine, and to explore using uttt.ai as a teacher for it.

**The rule change.** On CodinGame a full board with no three-in-a-row goes to
the player who won more subgames; only equal counts are a draw. The original
uttt.ai calls every full board a draw. `utttpy/game/ultimate_tic_tac_toe.py`
now scores it by count (see the commit). `utttcpp/` still uses the original
rules and is not used here.

Everything in `cg/` is tooling around that: a batched GPU self-play and
training loop, SPRT promotion gates, crossfish matches, a live dashboard and
game viewer, and a game-analysis script.

## Results so far

**Training.** Starting from uttt.ai's `policy_value_net_stage2`, fine-tuned on
self-play under CodinGame rules (5,000 games of 800 simulations per
generation). Each row is the new network against the previous best at 1,000
simulations per move:

| Network | Gate | Elo vs previous best |
|---|---|---|
| net1 | fixed 400 games vs original | +220 ± 36 |
| net2 | fixed 400 games | +51 ± 28 |
| net3 | fixed gate +16.5 ± 27; promoted by an early SPRT that counted pairs in completion order (biased) | +16.5 ± 27 |
| **net4** (best) | SPRT, pairs in opening order | **+22.3 ± 15** (LLR +3.36) |
| net5 | SPRT stopped by hand, undecided | +6.2 ± 10 (LLR +0.54) |

About +310 Elo over the original network in total, with gains shrinking
fast by net5.

**Against crossfish** (net4; equal wall time per move, one core each, one game
at a time, colour-swapped openings from crossfish's SPRT book, CodinGame
rules; crossfish's Elo):

| Time per move | Games | crossfish W/D/L | crossfish Elo |
|---|---|---|---|
| 90 ms | 1,000 | 581 / 174 / 245 | +121 ± 21 |
| 1 s | 500 | 222 / 141 / 137 | +60 ± 26 |
| 5 s | 100 | 29 / 49 / 22 | +24 ± 49 |

Run the matches one game at a time. Parallel games slow uttt.ai far more than
crossfish, because uttt.ai streams its 20 MB network from memory for every
simulation: at 8 parallel games uttt.ai ran at 25% of its single-game speed
against crossfish's 56%, which inflated crossfish's 90 ms result to +163
(`speed_probe.py` measures this).

**What the 90 ms games show** (`analysis/analyze.py`, served at `/analysis`):
the two engines judge early positions equally well; crossfish's search starts
proving results around ply 34. uttt.ai failed to convert 56 positions that
crossfish had proven lost (verified by re-search), worth about 33 Elo. It is
also weak on full-board count endings: crossfish scored 163-174-28 in those.

## Setup

Paths in the scripts assume the crossfish checkout at `C:/Users/natha/crossfish`,
whose `toolchains/` holds:

- `py312-dml`: Python 3.12 with `torch-directml` (training, torch self-play)
  and `onnxruntime-directml` (the fast self-play backend);
- `llvm-mingw-*`: the clang toolchain for building crossfish.

A Python 3.13 with CPU `onnxruntime` and `numpy` runs the crossfish matches,
the dashboard and the analysis.

1. Download the original network into `cg/`:
   `https://www.uttt.ai/policy_value_net_stage2.onnx`
   (19,983,650 bytes).
2. Fine-tuned networks: see this fork's GitHub releases (`net4.onnx`,
   `net4.pt`). Put them in `cg/train/`.
3. Build the crossfish binaries the match tools use (the plain bot, the
   metadata build and the debug build): `python cg/tools/make_crossfish_variants.py`.

## Layout

| Path | What it is |
|---|---|
| `utttai_onnx.py` | uttt.ai's NMCTS on CPU onnxruntime, batch 1 (the match player) |
| `bench_vs_utttai.py` | crossfish vs uttt.ai match driver; `NNNms` budgets give uttt.ai a time limit instead of a simulation count; writes a live move-by-move feed to `<log>.live/` |
| `run_bench_net4.sh` | the 90 ms / 1 s / 5 s series above |
| `speed_probe.py` | engine speed vs number of parallel games |
| `train/fused_net.py`, `train/onnx_weights.py` | the network with BatchNorm folded (as in the ONNX file), and reading/writing weights in place in the ONNX file |
| `train/selfplay.py` | batched self-play on the GPU (`--backend ort` is fastest), Dirichlet noise, temperature for 12 plies, CodinGame-rules results |
| `train/train.py` | fine-tuning: value target = mean of search value and game result; policy target = visit distribution plus an anchor to a reference network |
| `train/netmatch.py`, `train/sprt_gate.py` | network vs network: fixed match, and the pentanomial SPRT gate (H0 0, H1 +10 Elo, LLR ±3) |
| `train/loop.py` | the hill-climb: self-play, train, SPRT gate, stop after two failed promotions |
| `viewer/serve_viewer.py` | local server: dashboard `/`, game viewer `/games`, analysis `/analysis` |
| `analysis/analyze.py` | mines a recorded match for evaluation accuracy, disagreements, missed wins and style |
| `tools/make_crossfish_variants.py` | builds crossfish with search metadata (and a debug build) from the crossfish repo |

## Commands

From `cg/` unless noted; `PY_GPU` is `toolchains/py312-dml/Scripts/python.exe`.

```bash
# hill-climb from generation 6, starting from net4, up to generation 10 (in cg/train)
python loop.py 6 net4.pt 10

# one SPRT promotion gate (in cg/train)
$PY_GPU sprt_gate.py net5.pt net4.pt 1000 6 gate5.log

# crossfish vs net4, 1 s per move, 100 openings x 2 colours, one game at a time
UTTTAI_NET=train/net4.onnx python bench_vs_utttai.py 1000ms 1000 100 1 train/bench.log crossfish_cg_meta.exe

# dashboard and game viewer on http://localhost:8765
python viewer/serve_viewer.py 8765

# analyse a recorded match
python analysis/analyze.py train/bench_net4_90ms.live analysis/analysis_90ms.json
```
