"""Network Dissection core.

For a representation f(x) -> (B, K, h, w):
 1. T_k = top 0.5% quantile of unit k over all positions of all Broden images (exact)
 2. upsample A_k(x) to 112x112 (bilinear, anchored at receptive-field centres), M_k = S_k > T_k
 3. IoU_{k,c} = sum|M_k & L_c| / sum|M_k | L_c|, sums only over images having c's category
 4. unit k is a detector of its best concept if IoU > 0.04
All sums are kept per image shard so bootstrap intervals can be computed later.
"""
import math
import os
import time

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F


def rf_grid(h, w, start, jump, out_hw, in_hw, device):
    """grid_sample grid giving, for every label pixel centre, its position in activation units."""
    r = in_hw / out_hw
    u = ((torch.arange(out_hw, device=device, dtype=torch.float32) * r + (r - 1) / 2.0) - start) / jump
    gy, gx = torch.meshgrid(u / max(h - 1, 1) * 2 - 1, u / max(w - 1, 1) * 2 - 1, indexing="ij")
    return torch.stack([gx, gy], dim=-1)[None]


def upsample_rf(a, grid):
    return F.grid_sample(a, grid.expand(a.shape[0], -1, -1, -1), mode="bilinear",
                         padding_mode="border", align_corners=True)


@torch.no_grad()
def quantile_thresholds(feature_fn, loader, layers, q_top=0.005, sign=1.0, sample_batches=40, margin=4.0,
                        device="cuda", log=print):
    """T_k = (n_top+1)-th largest activation of unit k, n_top = ceil(q_top * #activations), so that
    P(a > T) = q_top exactly (ties give <= q_top, e.g. T = 0 for units that are almost always 0).
    Memory stays small: (A) a subsample gives some cut levels per unit, (B1) one pass counts how many
    values are > and == each level, (B2) a second pass keeps only values above the chosen level."""
    # candidate cut levels just above the target quantile; finer steps near it keep memory small for huge layers
    margins = torch.tensor([1.05, 1.1, 1.2, 1.4, 1.7] + [2.0 ** i for i in range(1, 9)]) * (margin / 4.0)
    margins = margins[margins * q_top < 0.9]
    J = len(margins)

    def flat(feats, l):                     # (K, B*h*w)
        f = (sign * feats[l]).float()
        return f.transpose(0, 1).reshape(f.shape[1], -1)

    samples = {l: [] for l in layers}
    for imgs, _ in loader.iter(subsample_batches=sample_batches):
        feats = feature_fn(imgs.to(device, non_blocking=True))
        for l in layers:
            fk = flat(feats, l)
            samples[l].append(fk[:, :: max(1, fk.shape[1] // 50_000)].cpu())
    levels = {}
    for l in layers:
        s = torch.cat(samples[l], 1)
        n = s.shape[1]
        ks = [min(n, max(1, int((1 - m * q_top) * (n - 1)) + 1)) for m in margins.tolist()]
        levels[l] = torch.stack([torch.kthvalue(s, k, dim=1).values for k in ks], 1).to(device)
    del samples

    gt = {l: torch.zeros(levels[l].shape, dtype=torch.int64, device=device) for l in layers}
    eq = {l: torch.zeros(levels[l].shape, dtype=torch.int64, device=device) for l in layers}
    total = {l: 0 for l in layers}
    for imgs, _ in loader.iter():
        feats = feature_fn(imgs.to(device, non_blocking=True))
        for l in layers:
            fk = flat(feats, l)
            total[l] += fk.shape[1]
            for j in range(J):
                gt[l][:, j] += (fk > levels[l][:, j:j + 1]).sum(1)
                eq[l][:, j] += (fk == levels[l][:, j:j + 1]).sum(1)

    out, need, sel = {}, {}, {}
    for l in layers:
        K = levels[l].shape[0]
        n_top = max(1, int(math.ceil(q_top * total[l])))
        ok = (gt[l] + eq[l]) >= n_top + 1
        if not bool(ok[:, -1].all()):
            raise RuntimeError(f"{l}: {int((~ok[:, -1]).sum())} units not bracketed")
        j_sel = ok.float().argmax(1)                       # highest level that has enough values
        sel[l] = levels[l][torch.arange(K), j_sel]
        gt_sel = gt[l][torch.arange(K), j_sel]
        need[l] = gt_sel >= n_top + 1                      # if not, T is exactly the level (tie)
        out[l] = sel[l].clone()
        mem = float(gt_sel[need[l]].sum()) * 12 / 2**30
        log(f"[quantile] {l}: n={total[l]}/unit, n_top={n_top}, tie units={int((~need[l]).sum())}, "
            f"candidate memory {mem:.2f} GB")
        if mem > float(os.environ.get("ND_CAND_GB", "48")):
            raise RuntimeError(f"{l}: candidate set too large ({mem:.1f} GB)")

    if any(bool(need[l].any()) for l in layers):
        cands = {l: [] for l in layers}
        for imgs, _ in loader.iter():
            feats = feature_fn(imgs.to(device, non_blocking=True))
            for l in layers:
                if need[l].any():
                    fk = flat(feats, l)
                    keep = (fk > sel[l][:, None]) & need[l][:, None]
                    cands[l].append((torch.nonzero(keep, as_tuple=True)[0].to(torch.int32).cpu(), fk[keep].cpu()))
        for l in layers:
            if not cands[l]:
                continue
            u = torch.cat([c[0] for c in cands[l]]).long()
            v = torch.cat([c[1] for c in cands[l]])[torch.argsort(u, stable=True)]
            n_top = max(1, int(math.ceil(q_top * total[l])))
            counts = torch.bincount(u, minlength=sel[l].shape[0])
            offs = torch.cumsum(counts, 0) - counts
            for k in torch.nonzero(need[l].cpu()).flatten().tolist():
                out[l][k] = torch.topk(v[offs[k]: offs[k] + counts[k]].to(device), n_top + 1).values[-1]
    return {l: t.cpu() for l, t in out.items()}, total


class IoUAccumulator:
    """Per-shard sums for IoU (paper Eq. 1)."""

    def __init__(self, K, n_labels, n_cats, n_shards, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.inter, self.mask_by_cat = z(n_shards, n_labels, K), z(n_shards, n_cats, K)
        self.label_px, self.mask_px = z(n_shards, n_labels), z(n_shards, K)

    def add(self, shard, M, seg, img_lab, cat_present):
        """M (B,K,H,W) bool masks, seg (B,L,H,W) pixel labels, img_lab (B,C) scene/texture labels
        (= full-image masks), cat_present (B,ncat)."""
        B, K, H, W = M.shape
        Mf = M.reshape(B, K, H * W).float()
        msum = Mf.sum(2)
        Mt = Mf.permute(0, 2, 1).reshape(B * H * W, K)
        n_labels = self.inter.shape[1]
        acc = torch.zeros(n_labels, K, device=M.device, dtype=torch.float32)
        for l in range(seg.shape[1]):
            acc.index_add_(0, seg[:, l].reshape(-1), Mt)    # counts per batch < 2^24, so float32 is exact
        acc += img_lab.float().t() @ msum                   # full-image label: intersection = mask size
        acc[0] = 0
        self.inter[shard] += acc.double()
        self.mask_by_cat[shard] += (cat_present.float().t() @ msum).double()
        lp = torch.bincount(seg.reshape(-1), minlength=n_labels)[:n_labels].double()
        lp += img_lab.double().sum(0) * H * W
        lp[0] = 0
        self.label_px[shard] += lp
        self.mask_px[shard] += msum.sum(0).double()

    def to_numpy(self):
        return {k: getattr(self, k).cpu().numpy() for k in ("inter", "mask_by_cat", "label_px", "mask_px")}


@torch.no_grad()
def accumulate_iou(feature_fn, loader, layers, thresholds, geometry, meta, n_shards, sign=1.0,
                   device="cuda", log=print):
    accs, grids = {}, {}
    t0 = time.time()
    for bi, (imgs, (seg, img_lab, cat_present, shard)) in enumerate(loader.iter(with_labels=True)):
        seg = seg.to(device, non_blocking=True).long()
        img_lab, cat_present = img_lab.to(device, non_blocking=True), cat_present.to(device, non_blocking=True)
        feats = feature_fn(imgs.to(device, non_blocking=True))
        for l in layers:
            f = (sign * feats[l]).float()
            B, K, h, w = f.shape
            if l not in accs:
                accs[l] = IoUAccumulator(K, meta["n_labels"], meta["n_cats"], n_shards, device)
                grids[l] = rf_grid(h, w, *geometry[l], seg.shape[-1], imgs.shape[1], device)
            M = upsample_rf(f, grids[l]) > thresholds[l].to(device).view(1, K, 1, 1)
            accs[l].add(shard, M, seg, img_lab, cat_present)
        if bi % 200 == 0:
            log(f"[iou] batch {bi} ({time.time() - t0:.0f}s)")
    return {l: a.to_numpy() for l, a in accs.items()}


def summarize(sums, meta, iou_thresh=0.04, shard_idx=None):
    """IoU matrix (K,C), best concept of each unit and whether it is a detector."""
    sel = slice(None) if shard_idx is None else shard_idx
    inter = sums["inter"][sel].sum(0)
    mk = sums["mask_by_cat"][sel].sum(0)[meta["label_cat"]]       # |M_k| over images with cat(c)
    union = mk + sums["label_px"][sel].sum(0)[:, None] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).T
    iou[:, ~meta["label_valid"]] = 0.0
    best = iou.argmax(1)
    best_iou = iou[np.arange(iou.shape[0]), best]
    return {"iou": iou, "best": best, "best_iou": best_iou, "is_detector": best_iou > iou_thresh}


def unique_detectors(s, meta):
    det = s["best"][s["is_detector"]]
    uniq = np.unique(det)
    lc = meta["label_cat"]
    return {"unique_total": int(len(uniq)),
            "unique_per_cat": {c: int(np.sum(lc[uniq] == i)) for i, c in enumerate(meta["cat_names"])},
            "detector_units": int(s["is_detector"].sum()),
            "detector_units_per_cat": {c: int(np.sum(lc[det] == i)) for i, c in enumerate(meta["cat_names"])}}


# ---- random rotations of a representation (paper Sec. 3.2) ----
def random_rotation(n, seed=0):
    """Uniform random Q in SO(n): QR of a Gaussian matrix with sign fix, det forced to +1."""
    A = np.random.default_rng(seed).standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def fractional_rotation(Q, alpha):
    """Q^alpha on the geodesic from I to Q: real Schur form, every rotation angle times alpha."""
    if alpha == 0:
        return np.eye(Q.shape[0])
    if alpha == 1:
        return Q.copy()
    T, U = scipy.linalg.schur(Q, output="real")
    n, Ta, i = Q.shape[0], np.zeros_like(T), 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-10:      # 2x2 rotation block
            th = np.arctan2(T[i + 1, i], T[i, i]) * alpha
            Ta[i:i + 2, i:i + 2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
            i += 2
        else:
            if T[i, i] < 0:
                raise ValueError("eigenvalue -1 without a partner")
            Ta[i, i] = 1.0
            i += 1
    return U @ Ta @ U.T
