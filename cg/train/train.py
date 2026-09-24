"""Fine-tune uttt.ai's network on CodinGame-rules self-play.

Targets per position:
  policy: the search's visit distribution (uttt.ai's masked KL), weight
          1 - anchor_weight;
  anchor: an anchor network's policy (the original uttt.ai net, or the
          current best), weight anchor_weight, so low-simulation visit targets
          cannot drag the policy far from a trusted one in one step;
  value:  0.5 * search value + 0.5 * game result under CodinGame rules
          (uttt.ai used the search value alone), weight W_VALUE (uttt.ai: 3.0).
Batches get a random board symmetry (row flip, column flip, transpose), as in
uttt.ai's random_orientation_inplace. The newest data directory's first shard
is held out for validation.

usage: train.py <init: .onnx or .pt> <data_dir[,data_dir...]> <out_prefix> [epochs=4] [lr=1e-4]
                [anchor=original .onnx or a .pt] [anchor_weight=0.5]
"""
import glob
import math
import sys
import time
import warnings

import numpy as np
import torch

import fused_net
import selfplay
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe

warnings.filterwarnings("ignore")
W_VALUE = 3.0
BATCH = 1024
ORIGINAL = "../policy_value_net_stage2.onnx"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(dirs):
    """Shards in directory order; the newest directory's first shard comes first (validation)."""
    per_dir = [sorted(glob.glob(f"{d}/*.npz")) for d in dirs]
    files = per_dir[-1][:1] + [f for fs in per_dir[:-1] for f in fs] + per_dir[-1][1:]
    return files, [np.load(f) for f in files]


def build(shards):
    states = np.concatenate([s["state"] for s in shards])
    policy_idx = np.concatenate([s["policy"] for s in shards]).astype(np.float32)
    value = 0.5 * np.concatenate([s["q"] for s in shards]) + 0.5 * np.concatenate([s["z"] for s in shards])
    z = np.concatenate([s["z"] for s in shards])
    n = len(states)
    x = np.zeros((n, 4, 9, 9), dtype=np.int8)
    pol = np.zeros((n, 9, 9), dtype=np.float32)
    for i in range(n):
        u = UltimateTicTacToe(state=bytearray(states[i].tobytes()))
        x[i] = selfplay.encode(u, u.get_legal_indexes()).astype(np.int8)
    pol[:, selfplay.ROW, selfplay.COL] = policy_idx
    return x, pol, value.astype(np.float32), z.astype(np.float32)


def orient(t, flip_r, flip_c, transpose, spatial=(2, 3)):
    r, c = spatial
    if flip_r:
        t = t.flip(r)
    if flip_c:
        t = t.flip(c)
    if transpose:
        t = t.transpose(r, c)
    return t


def masked_logp(logits, mask):
    return torch.log_softmax(logits.masked_fill(~mask, -1e9).flatten(1), dim=1)


def losses(net, x, pol, anchor_p, value, dev):
    mask = x[:, 3] > 0.5
    logits, v = net(x)
    logp = masked_logp(logits, mask)
    ce = -(pol.flatten(1) * logp).sum(1).mean()
    ce_anchor = -(anchor_p.flatten(1) * logp).sum(1).mean()
    mse = ((v - value) ** 2).mean()
    return ce, ce_anchor, mse


def main():
    import torch_directml
    init, dirs, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    lr_max = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
    anchor_path = sys.argv[6] if len(sys.argv) > 6 else ORIGINAL
    w_anchor = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
    w_policy = 1.0 - w_anchor
    dev = torch_directml.device()

    files, shards = load_rows(dirs)
    val_shards, train_shards = shards[:1], shards[1:]
    log(f"{len(files)} shards; building inputs")
    tr = build(train_shards)
    va = build(val_shards)
    log(f"train {len(tr[0]):,} positions, validation {len(va[0]):,} (held-out shard {files[0]})")

    # Anchor targets: the anchor network's policy on every position.
    def load_net(path):
        if str(path).endswith(".onnx"):
            return fused_net.from_onnx(path)
        m = fused_net.FusedPolicyValueNet()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return m

    anchor_net = load_net(anchor_path).to(dev).eval()
    log(f"anchor {anchor_path} weight {w_anchor}; policy weight {w_policy}")

    def anchor_probs(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                xb = torch.from_numpy(x[i:i + 4096]).float().to(dev)
                out.append(torch.softmax(anchor_net(xb)[0].masked_fill(xb[:, 3] < 0.5, -1e9).flatten(1), 1).view(-1, 9, 9).cpu())
        return torch.cat(out).numpy()

    tr_anchor, va_anchor = anchor_probs(tr[0]), anchor_probs(va[0])

    net = load_net(init).to(dev)

    def evaluate(model, data, anchor):
        model.eval()
        tot = np.zeros(4)
        n = 0
        with torch.no_grad():
            for i in range(0, len(data[0]), 4096):
                xb = torch.from_numpy(data[0][i:i + 4096]).float().to(dev)
                pb = torch.from_numpy(data[1][i:i + 4096]).to(dev)
                ab = torch.from_numpy(anchor[i:i + 4096]).to(dev)
                vb = torch.from_numpy(data[2][i:i + 4096]).to(dev)
                zb = torch.from_numpy(data[3][i:i + 4096]).to(dev)
                ce, cea, mse = losses(model, xb, pb, ab, vb, dev)
                v = model(xb)[1]
                zmse = ((v - zb) ** 2).mean()
                k = len(xb)
                tot += np.array([ce.item(), cea.item(), mse.item(), zmse.item()]) * k
                n += k
        model.train()
        return tot / n

    def report(tag, m):
        log(f"{tag}: policy CE {m[0]:.4f}  anchor CE {m[1]:.4f}  value MSE {m[2]:.4f}  MSE vs game result {m[3]:.4f}")

    report("validation, anchor network", evaluate(anchor_net, va, va_anchor))
    del anchor_net

    opt = torch.optim.Adam(net.parameters(), lr=lr_max)
    n = len(tr[0])
    steps = epochs * (n // BATCH)
    warm = min(200, steps // 10)
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        run = np.zeros(3)
        for b in range(n // BATCH):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            lr = lr_max * (step + 1) / warm if step < warm else \
                1e-5 + 0.5 * (lr_max - 1e-5) * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            fr, fc, tp = np.random.randint(2, size=3)
            xb = orient(torch.from_numpy(tr[0][idx]).float().to(dev), fr, fc, tp)
            pb = orient(torch.from_numpy(tr[1][idx]).to(dev), fr, fc, tp, (1, 2))
            ab = orient(torch.from_numpy(tr_anchor[idx]).to(dev), fr, fc, tp, (1, 2))
            vb = torch.from_numpy(tr[2][idx]).to(dev)
            ce, cea, mse = losses(net, xb, pb, ab, vb, dev)
            loss = w_policy * ce + w_anchor * cea + W_VALUE * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += [ce.item(), cea.item(), mse.item()]
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                eta = time.strftime("%H:%M", time.localtime(t0 + el / step * steps))
                r = run / (b + 1)
                log(f"epoch {ep + 1}/{epochs} step {step}/{steps} lr {lr:.1e}  "
                    f"policy CE {r[0]:.4f} anchor CE {r[1]:.4f} value MSE {r[2]:.4f}  ETA {eta}")
        report(f"validation after epoch {ep + 1}", evaluate(net, va, va_anchor))
    net = net.cpu()
    torch.save(net.state_dict(), f"{out}.pt")
    fused_net.to_onnx(net, ORIGINAL, f"{out}.onnx")
    log(f"saved {out}.pt and {out}.onnx")


if __name__ == "__main__":
    main()
