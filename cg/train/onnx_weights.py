"""Minimal ONNX reader: graph nodes and initializer tensors, no onnx package.

Only the protobuf fields this model uses are decoded (ModelProto.graph,
GraphProto.node/initializer, NodeProto input/output/op_type/attribute,
TensorProto dims/data_type/raw_data/float_data).
"""
import struct
from dataclasses import dataclass, field

import numpy as np


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for one message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]
            i += 8
        elif wt == 2:
            n, i = _varint(buf, i)
            v = buf[i:i + n]
            i += n
        elif wt == 5:
            v = buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, wt, v


@dataclass
class Node:
    op_type: str = ""
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


def _attribute(buf):
    name, ints, value = "", [], None
    for num, wt, v in _fields(buf):
        if num == 1:
            name = bytes(v).decode()
        elif num == 2:
            value = struct.unpack("<f", v)[0]
        elif num == 3:
            value = v if v < (1 << 63) else v - (1 << 64)
        elif num == 8:
            if wt == 2:  # packed
                j = 0
                while j < len(v):
                    x, j = _varint(v, j)
                    ints.append(x if x < (1 << 63) else x - (1 << 64))
            else:
                ints.append(v)
    return name, (ints if ints else value)


def _tensor(buf):
    dims, dtype, name, raw, floats = [], 0, "", None, []
    for num, wt, v in _fields(buf):
        if num == 1:
            if wt == 2:
                j = 0
                while j < len(v):
                    x, j = _varint(v, j)
                    dims.append(x)
            else:
                dims.append(v)
        elif num == 2:
            dtype = v
        elif num == 8:
            name = bytes(v).decode()
        elif num == 9:
            raw = bytes(v)
        elif num == 4:
            floats += list(struct.unpack(f"<{len(v) // 4}f", v)) if wt == 2 else [struct.unpack("<f", v)[0]]
    if dtype == 1:
        arr = np.frombuffer(raw, dtype="<f4") if raw is not None else np.array(floats, dtype=np.float32)
    elif dtype == 7:
        arr = np.frombuffer(raw, dtype="<i8")
    else:
        raise ValueError(f"unsupported tensor dtype {dtype} for {name}")
    return name, arr.reshape(dims).copy()


def load(path):
    model = open(path, "rb").read()
    graph = next(v for num, wt, v in _fields(model) if num == 7)
    nodes, weights = [], {}
    for num, wt, v in _fields(graph):
        if num == 1:
            n = Node()
            for fnum, _, fv in _fields(v):
                if fnum == 1:
                    n.inputs.append(bytes(fv).decode())
                elif fnum == 2:
                    n.outputs.append(bytes(fv).decode())
                elif fnum == 4:
                    n.op_type = bytes(fv).decode()
                elif fnum == 5:
                    k, val = _attribute(fv)
                    n.attrs[k] = val
            nodes.append(n)
        elif num == 5:
            name, arr = _tensor(v)
            weights[name] = arr
    return nodes, weights


def _fields_at(buf, base):
    """Like _fields, but also yields the absolute offset of each value."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 2:
            n, i = _varint(buf, i)
            yield num, base + i, buf[i:i + n]
            i += n
        elif wt == 0:
            _, i = _varint(buf, i)
        elif wt == 1:
            i += 8
        elif wt == 5:
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


def raw_data_offsets(path):
    """name -> (offset, nbytes) of each float initializer's raw_data in the file."""
    model = open(path, "rb").read()
    out = {}
    for num, gbase, graph in _fields_at(model, 0):
        if num != 7:
            continue
        for gnum, tbase, tensor in _fields_at(graph, gbase):
            if gnum != 5:
                continue
            name, raw = None, None
            for tnum, off, v in _fields_at(tensor, tbase):
                if tnum == 8:
                    name = bytes(v).decode()
                elif tnum == 9:
                    raw = (off, len(v))
            if name is not None and raw is not None:
                out[name] = raw
    return out


def write_weights(src, dst, new_weights):
    """Copy src to dst with the named float32 initializers replaced (same shapes)."""
    data = bytearray(open(src, "rb").read())
    offsets = raw_data_offsets(src)
    for name, arr in new_weights.items():
        off, nbytes = offsets[name]
        blob = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        assert len(blob) == nbytes, (name, len(blob), nbytes)
        data[off:off + nbytes] = blob
    open(dst, "wb").write(bytes(data))


def _decode(buf):
    """Message -> ordered list of (field, wire_type, value); value is int or bytes."""
    items, i = [], 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = bytes(buf[i:i + 8]), i + 8
        elif wt == 2:
            n, i = _varint(buf, i)
            v, i = bytes(buf[i:i + n]), i + n
        elif wt == 5:
            v, i = bytes(buf[i:i + 4]), i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        items.append((num, wt, v))
    return items


def _enc_varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _encode(items):
    out = bytearray()
    for num, wt, v in items:
        out += _enc_varint((num << 3) | wt)
        if wt == 0:
            out += _enc_varint(v)
        elif wt == 2:
            out += _enc_varint(len(v)) + v
        else:
            out += v
    return bytes(out)


def _edit(buf, path, fn):
    """Apply fn to every submessage reached by following field numbers in path."""
    items = _decode(buf)
    out = []
    for num, wt, v in items:
        if wt == 2 and path and num in path[0]:
            v = fn(v) if len(path) == 1 else _edit(v, path[1:], fn)
        out.append((num, wt, v))
    return _encode(out)


def set_dynamic_batch(src, dst, name="batch"):
    """Make dim 0 of every graph input and output symbolic (any batch size)."""
    def first_dim_symbolic(shape):
        items = _decode(shape)
        for k, (num, wt, v) in enumerate(items):
            if num == 1:  # first TensorShapeProto.dim
                items[k] = (1, 2, _encode([(2, 2, name.encode())]))  # Dimension.dim_param
                break
        return _encode(items)
    # ModelProto.graph(7) -> input(11)/output(12) -> type(2) -> tensor_type(1) -> shape(2)
    model = open(src, "rb").read()
    open(dst, "wb").write(_edit(model, [{7}, {11, 12}, {2}, {1}, {2}], first_dim_symbolic))


if __name__ == "__main__":
    import sys
    nodes, weights = load(sys.argv[1])
    print(f"{len(nodes)} nodes, {len(weights)} initializers, "
          f"{sum(w.size for w in weights.values()):,} parameters")
    for n in nodes:
        shapes = [tuple(weights[i].shape) for i in n.inputs if i in weights]
        print(f"{n.op_type:16s} {shapes} {n.attrs if n.op_type in ('Conv', 'ConvTranspose') else ''}")
