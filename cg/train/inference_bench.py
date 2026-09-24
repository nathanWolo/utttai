"""Inference-only comparison on an idle GPU: torch-DirectML (fp32, fp16) vs
onnxruntime DirectML, checked for agreement first, then timed at several batch
sizes with interleaved repetitions (median of 7).

usage: inference_bench.py <net.onnx>
"""
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import selfplay  # noqa: E402

net = sys.argv[1]
ref = np.load("ort_reference.npz")
evs = {
    "torch fp32": selfplay.Evaluator(net, "torch"),
    "torch fp16": selfplay.Evaluator(net, "torch", fp16=True),
    "ort DirectML fp32": selfplay.Evaluator(net, "ort"),
}
x = list(ref["x"].astype(np.float32))
base_pl, base_v = evs["torch fp32"](x)
for k, ev in evs.items():
    pl, v = ev(x)
    agree = (pl.reshape(len(x), -1).argmax(1) == base_pl.reshape(len(x), -1).argmax(1)).mean()
    print(f"{k:18s} vs torch fp32: max value diff {np.abs(v - base_v).max():.5f}, "
          f"max logit diff {np.abs(pl - base_pl).max():.4f}, top-move agreement {agree:.3f}")

for bs in (64, 128, 256, 512, 1024):
    xs = [x[i % len(x)] for i in range(bs)]
    times = {k: [] for k in evs}
    for rep in range(7):
        for k, ev in evs.items():
            ev(xs)
            t = time.perf_counter()
            for _ in range(10):
                ev(xs)
            times[k].append((time.perf_counter() - t) / 10)
    base = np.median(times["torch fp32"])
    print(f"batch {bs:5d}: " + " | ".join(
        f"{k} {np.median(v) * 1000:6.2f} ms ({bs / np.median(v):,.0f}/s, {base / np.median(v):.2f}x)"
        for k, v in times.items()), flush=True)
