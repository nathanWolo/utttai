"""uttt.ai's PolicyValueNetwork (Apache-2.0) with BatchNorm folded into the
preceding convolutions, exactly as in the deployed policy_value_net_stage2.onnx.
The auxiliary action-values head is absent from the ONNX export and is omitted.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import onnx_weights


def elu(x):
    # ELU (alpha=1) from ops DirectML runs natively; torch's elu falls back to the CPU there.
    return torch.clamp(x, min=0) + torch.exp(torch.clamp(x, max=0)) - 1


class FusedPolicyValueNet(nn.Module):
    def __init__(self, planes: int = 256):
        super().__init__()
        h, p2 = planes // 2, planes * 2
        self.conv1 = nn.Conv2d(4, h, 3, stride=3)
        self.c2a = nn.Conv2d(h, h, 1)
        self.c2b = nn.Conv2d(h, planes, 1)
        self.blocks = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(planes, p2, 1), nn.Conv2d(p2, p2, 3, groups=p2),
                           nn.ConvTranspose2d(p2, planes, 3)])
            for _ in range(3)
        ])
        self.p_conv = nn.Conv2d(5 * planes // 2, planes, 1)
        self.p_up = nn.ConvTranspose2d(planes, planes, 3, stride=3)
        self.p_l1 = nn.Conv2d(4 + planes, 128, 1)
        self.p_l2 = nn.Conv2d(128, 1, 1)
        self.v1a = nn.Conv2d(planes, planes, 1)
        self.v1b = nn.Conv2d(planes, planes, 3, groups=planes)
        self.v2a = nn.Conv2d(planes, planes, 1)
        self.v2b = nn.Conv2d(planes, planes, 3, groups=planes)
        self.fc1 = nn.Linear(2 * planes, 128)
        self.fc2 = nn.Linear(128, 1)
        self.planes = planes

    def weighted_layers(self):
        """Layers in the order their weights appear in the ONNX graph."""
        out = [self.conv1, self.c2a, self.c2b]
        for b in self.blocks:
            out += list(b)
        out += [self.p_conv, self.p_up, self.p_l1, self.p_l2,
                self.v1a, self.v1b, self.v2a, self.v2b, self.fc1, self.fc2]
        return out

    def forward(self, x):
        x1 = elu(self.conv1(x))
        x2 = torch.relu(self.c2b(elu(self.c2a(x1))))
        x3 = x2
        for a, dw, up in self.blocks:
            x3 = torch.relu(up(elu(dw(elu(a(x3))))) + x3)
        p = elu(self.p_conv(torch.cat([x1, x2, x3], dim=1)))
        p = elu(self.p_up(p))
        p = torch.cat([x, p], dim=1)
        logits = self.p_l2(elu(self.p_l1(p))).clamp(-32, 32).squeeze(1)
        b = x.size(0)
        v1 = elu(self.v1b(self.v1a(x2))).view(b, self.planes)
        v2 = elu(self.v2b(self.v2a(x3))).view(b, self.planes)
        v = torch.tanh(self.fc2(elu(self.fc1(torch.cat([v1, v2], dim=1))))).squeeze(1)
        return logits, v


def from_onnx(path) -> FusedPolicyValueNet:
    nodes, weights = onnx_weights.load(str(path))
    net = FusedPolicyValueNet()
    tensors = []
    for n in nodes:
        if n.op_type in ("Conv", "ConvTranspose", "Gemm"):
            w, bias = weights[n.inputs[1]], weights[n.inputs[2]]
            if n.op_type == "Gemm" and not n.attrs.get("transB", 0):
                w = w.T
            tensors.append((w, bias))
    layers = net.weighted_layers()
    assert len(tensors) == len(layers), (len(tensors), len(layers))
    with torch.no_grad():
        for layer, (w, bias) in zip(layers, tensors):
            assert tuple(layer.weight.shape) == w.shape, (layer, w.shape)
            layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
            layer.bias.copy_(torch.from_numpy(bias))
    return net


def to_onnx(net: FusedPolicyValueNet, template, dst) -> None:
    """Write net's weights into a copy of the template ONNX (identical graph)."""
    nodes, weights = onnx_weights.load(str(template))
    names = [(n.inputs[1], n.inputs[2], n.op_type, n.attrs.get("transB", 0))
             for n in nodes if n.op_type in ("Conv", "ConvTranspose", "Gemm")]
    new = {}
    for layer, (wname, bname, op, trans_b) in zip(net.weighted_layers(), names):
        w = layer.weight.detach().cpu().numpy()
        if op == "Gemm" and not trans_b:
            w = w.T
        new[wname] = w
        new[bname] = layer.bias.detach().cpu().numpy()
    onnx_weights.write_weights(str(template), str(dst), new)
