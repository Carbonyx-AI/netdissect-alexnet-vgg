"""Load Caffe models (.prototxt + .caffemodel) into PyTorch, without Caffe.

The paper's models are Caffe models. .caffemodel files are Google protobuf messages; we read the
wire format directly (only the few fields we need), and build the network from the .prototxt.
Caffe details handled: grouped convolutions, LRN, pooling with ceil rounding, BGR 0-255 input with
mean subtraction. Supports old (V1, "layers") and new ("layer") formats.
"""
import re

import numpy as np
import torch
import torch.nn as nn


# ---------------- protobuf wire format ----------------
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b):
    """yield (field number, wire type, value) of one message."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, t = key >> 3, key & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 1:
            v, i = b[i:i + 8], i + 8
        elif t == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif t == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {t}")
        yield f, t, v


def read_blob(b):
    """BlobProto -> numpy array (data=5 floats, shape=7 or num/channels/height/width=1..4)."""
    data, shape, old = [], None, {}
    for f, t, v in _fields(b):
        if f == 5:
            data.append(np.frombuffer(v, "<f4") if t == 2 else np.frombuffer(v, "<f4", 1))
        elif f == 7:
            shape = []
            for ff, tt, vv in _fields(v):
                if ff == 1:
                    if tt == 2:                 # packed
                        j = 0
                        while j < len(vv):
                            d, j = _varint(vv, j); shape.append(d)
                    else:
                        shape.append(vv)
        elif f in (1, 2, 3, 4) and t == 0:
            old[f] = v
    arr = np.concatenate(data) if data else np.zeros(0, np.float32)
    if shape is None:
        shape = [old.get(k, 1) for k in (1, 2, 3, 4)]
    return arr.reshape(shape)


def read_caffemodel(path):
    """{layer name: [blobs]} for V1 ('layers', field 2) and new ('layer', field 100) nets."""
    b = open(path, "rb").read()
    out = {}
    for f, t, v in _fields(b):
        if f in (2, 100) and t == 2:
            name_f, blob_f = (4, 6) if f == 2 else (1, 7)
            name, blobs = None, []
            for ff, tt, vv in _fields(v):
                if ff == name_f and tt == 2:
                    name = vv.decode()
                elif ff == blob_f and tt == 2:
                    blobs.append(read_blob(vv))
            if blobs:
                out[name] = blobs
    return out


# ---------------- prototxt ----------------
def parse_prototxt(text):
    """tiny parser: returns list of layer dicts in order (type, name, params as flat dict)."""
    text = re.sub(r"#.*", "", text)
    toks = re.findall(r'"[^"]*"|[{}:]|[^\s{}:"]+', text)
    def block(i):
        d = {}
        while i < len(toks) and toks[i] != "}":
            key = toks[i]
            if toks[i + 1] == ":" and toks[i + 2] != "{":
                val = toks[i + 2].strip('"'); i += 3
            else:
                val, i = block(i + (3 if toks[i + 1] == ":" else 2)); i += 1
            d.setdefault(key, []).append(val)
        return d, i
    top, _ = block(0)
    layers = []
    for L in top.get("layer", []) + top.get("layers", []):
        flat = {"name": L["name"][0], "type": L["type"][0].upper()}
        for k, v in L.items():
            if k.endswith("_param") and isinstance(v[0], dict):
                for kk, vv in v[0].items():
                    if not isinstance(vv[0], dict):
                        flat[kk] = vv[0]
        layers.append(flat)
    return layers


class CaffeNet(nn.Module):
    """features (convs) + classifier (fully connected), same attribute names as torchvision models."""

    def __init__(self, layers, blobs, mean_bgr, in_size=224):
        super().__init__()
        feats, cls, ch = [], [], 3
        self.names = []
        for L in layers:
            t = L["type"]
            if t == "CONVOLUTION":
                k, s, p, g = (int(L.get(x, d)) for x, d in (("kernel_size", 1), ("stride", 1), ("pad", 0), ("group", 1)))
                conv = nn.Conv2d(ch, int(L["num_output"]), k, s, p, groups=g)
                w, bias = blobs[L["name"]]
                conv.weight.data = torch.from_numpy(w.reshape(conv.weight.shape).copy())
                conv.bias.data = torch.from_numpy(bias.reshape(-1).copy())
                feats.append(conv); ch = conv.out_channels
            elif t == "RELU":
                (cls if cls else feats).append(nn.ReLU())
            elif t == "POOLING":
                assert L.get("pool", "MAX") == "MAX"
                feats.append(nn.MaxPool2d(int(L["kernel_size"]), int(L.get("stride", 1)), int(L.get("pad", 0)), ceil_mode=True))
            elif t == "LRN":
                n = int(L.get("local_size", 5))
                feats.append(nn.LocalResponseNorm(n, float(L.get("alpha", 1)), float(L.get("beta", 0.75)), float(L.get("k", 1))))
            elif t == "INNER_PRODUCT" or t == "INNERPRODUCT":
                w, bias = blobs[L["name"]]
                if not cls:
                    with torch.no_grad():
                        n_in = nn.Sequential(*feats)(torch.zeros(1, 3, in_size, in_size)).numel()
                else:
                    n_in = w.size // int(L["num_output"])
                fc = nn.Linear(n_in, int(L["num_output"]))
                fc.weight.data = torch.from_numpy(w.reshape(fc.weight.shape).copy())
                fc.bias.data = torch.from_numpy(bias.reshape(-1).copy())
                cls.append(fc)
            elif t == "DROPOUT":
                cls.append(nn.Dropout(float(L.get("dropout_ratio", 0.5))))
        self.features, self.avgpool, self.classifier = nn.Sequential(*feats), nn.Identity(), nn.Sequential(*cls)
        self.register_buffer("mean_bgr", torch.tensor(mean_bgr, dtype=torch.float32).view(1, 3, 1, 1))

    def preprocess(self, x01_rgb):
        """RGB in [0,1] -> Caffe input: BGR, 0..255, mean subtracted."""
        return x01_rgb.flip(1) * 255.0 - self.mean_bgr

    def forward(self, x):            # x already preprocessed
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


def load_caffe(prototxt, caffemodel, mean_bgr):
    return CaffeNet(parse_prototxt(open(prototxt).read()), read_caffemodel(caffemodel), mean_bgr)


def mean_from_binaryproto(path):
    """per-channel BGR mean of a Caffe mean image."""
    m = read_blob(open(path, "rb").read())
    return m.reshape(-1, m.shape[-2], m.shape[-1]).mean((1, 2)).tolist()
