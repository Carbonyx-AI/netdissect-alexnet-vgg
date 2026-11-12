"""Checks of the core parts.  python tests/test_core.py"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netdissect import dissect as D  # noqa: E402
from netdissect.models import FeatureExtractor, load_model, rf_geometry  # noqa: E402


class Loader:            # tiny stand-in for the Broden loader
    def __init__(self, data):
        self.data = data

    def iter(self, subsample_batches=None, with_labels=False):
        for i in range(0, len(self.data), 8):
            yield self.data[i:i + 8], None


def test_inplace_relu_problem():
    """why we switch off in-place ReLU: it overwrites a tensor captured by a hook on the conv."""
    conv, relu, got = nn.Conv2d(1, 1, 1, bias=False), nn.ReLU(inplace=True), {}
    conv.weight.data.fill_(1.0)
    conv.register_forward_hook(lambda m, i, o: got.__setitem__("x", o))
    relu(conv(-torch.ones(1, 1, 2, 2)))
    assert got["x"].min() == 0
    fx = FeatureExtractor(load_model("alexnet-random"), "alexnet", ["conv5"], pre_relu=True)
    assert fx(torch.randn(2, 3, 224, 224))["conv5"].min() < 0


def test_rf_geometry():
    m = load_model("alexnet-random")
    assert rf_geometry(m, 0) == (3.0, 4.0)        # conv1: k=11, s=4, p=2
    assert rf_geometry(m, 11)[1] == 16.0         # conv5 stride


def test_quantile_exact_and_ties():
    torch.manual_seed(0)
    data = torch.randn(40, 3, 5, 5)
    data[:, 2] = torch.relu(torch.randn(40, 5, 5) - 3.0)      # almost always 0 -> T must be 0
    T, _ = D.quantile_thresholds(lambda x: {"l": x}, Loader(data), ["l"], q_top=0.01, device="cpu",
                                 sample_batches=2, log=lambda *a: None)
    for k in range(2):
        v = data[:, k].reshape(-1)
        assert int((v > T["l"][k]).sum()) == int(np.ceil(0.01 * v.numel()))
    assert T["l"][2] == 0.0


def test_iou_against_bruteforce():
    rng = np.random.default_rng(0)
    B, K, H, W, C = 6, 3, 8, 8, 5
    M = torch.from_numpy(rng.random((B, K, H, W)) > 0.6)
    seg = torch.from_numpy(rng.integers(0, C - 1, (B, 2, H, W)))      # label 4 = image-level only
    seg[:, 1][seg[:, 1] == seg[:, 0]] = 0
    img_lab = torch.zeros(B, C, dtype=torch.uint8)
    img_lab[0, 4] = img_lab[3, 4] = 1
    label_cat = np.array([0, 0, 0, 1, 2])
    cat = torch.tensor([[int(any(bool((seg[b] == c).any()) or bool(img_lab[b, c]) for c in range(1, C) if label_cat[c] == k))
                         for k in range(3)] for b in range(B)], dtype=torch.uint8)
    acc = D.IoUAccumulator(K, C, 3, 1, "cpu")
    acc.add(0, M, seg, img_lab, cat)
    iou = D.summarize(acc.to_numpy(), {"label_cat": label_cat, "label_valid": np.ones(C, bool)})["iou"]
    for k in range(K):
        for c in range(1, C):
            I = U = 0
            for b in range(B):
                if cat[b, label_cat[c]]:
                    Lc = (seg[b] == c).any(0) if c < 4 else torch.full((H, W), bool(img_lab[b, c]))
                    I += int((M[b, k] & Lc).sum())
                    U += int((M[b, k] | Lc).sum())
            assert abs(iou[k, c] - (I / U if U else 0)) < 1e-9


def test_rotation():
    Q = D.random_rotation(16, seed=1)
    assert np.allclose(Q @ Q.T, np.eye(16)) and np.linalg.det(Q) > 0
    H = D.fractional_rotation(Q, 0.5)
    assert np.allclose(H @ H, Q, atol=1e-8) and np.allclose(H @ H.T, np.eye(16), atol=1e-8)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
