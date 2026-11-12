"""Broden dataset: one-time preprocessing into memmaps, and a streaming loader.

Broden (broden1_224) files used:
  index.csv  image, split, ih, iw, sh, sw, color, object, part, material, scene, texture
             (pixel categories = ';'-separated PNG files, scene/texture = label numbers for full image)
  label.csv  number, name, category e.g. "object(123);part(4)", ...
  label PNG  label number = R + 256*G
Preprocessing writes images.npy (N,224,224,3) uint8, seg.npy (N,6,112,112) uint16 and meta.npz/json.
Run:  python -m netdissect.data --root work/broden1_224 --out work/broden
"""
import argparse
import json
import os
import queue
import re
import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image

PIXEL_CATS = ["color", "object", "part", "material"]
IMAGE_CATS = ["scene", "texture"]
CATS = ["object", "part", "scene", "material", "texture", "color"]   # same order as the paper


def parse_labels(label_csv):
    """names and primary category (the one with most images) of every label."""
    lab = pd.read_csv(label_csv)
    n_labels = int(lab["number"].max()) + 1
    names = [""] * n_labels
    primary = np.full(n_labels, -1, dtype=np.int64)
    for _, r in lab.iterrows():
        n = int(r["number"])
        names[n] = str(r["name"])
        best, best_cnt = None, -1
        for part in str(r["category"]).split(";"):
            m = re.match(r"(\w+)\((\d+)\)", part.strip())
            if m and int(m.group(2)) > best_cnt:
                best, best_cnt = m.group(1), int(m.group(2))
        if best is not None:
            primary[n] = CATS.index(best)
    return names, primary, n_labels


def decode_png(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int64)
    return a[..., 0] + a[..., 1] * 256


def read_one(args):
    """image + all its pixel label maps merged into L layers (no label twice on a pixel)."""
    i, row, root, L, sh, sw = args
    img = np.asarray(Image.open(os.path.join(root, "images", row["image"])).convert("RGB"), dtype=np.uint8)
    maps = [decode_png(os.path.join(root, "images", p))
            for c in PIXEL_CATS if row[c] for p in row[c].split(";")]
    seg = np.zeros((L, sh, sw), dtype=np.uint16)
    filled = 0
    for m in maps:
        assert m.shape == (sh, sw), (row["image"], m.shape)
        if filled == 0:
            seg[0] = m
            filled = 1
            continue
        dup = np.zeros_like(m, dtype=bool)
        for l in range(filled):
            dup |= seg[l] == m
        m = np.where(dup, 0, m)
        if (m > 0).any():
            for l in range(filled):          # put it in the first layer which is free there
                if not ((seg[l] > 0) & (m > 0)).any():
                    seg[l] = np.where(m > 0, m, seg[l])
                    break
            else:
                if filled >= L:
                    raise RuntimeError(f"more than {L} label layers needed for {row['image']}")
                seg[filled] = m
                filled += 1
    img_labels = [int(x) for c in IMAGE_CATS if row[c] for x in row[c].split(";")]
    return i, img, seg, img_labels, filled


def preprocess(root, out, workers=8, layers=6, shards=50, seed=0):
    os.makedirs(out, exist_ok=True)
    idx = pd.read_csv(os.path.join(root, "index.csv"), keep_default_na=False, dtype=str)
    N = len(idx)
    ih, iw, sh, sw = (int(idx[c].iloc[0]) for c in ("ih", "iw", "sh", "sw"))
    assert (idx["ih"].astype(int) == ih).all() and (idx["sh"].astype(int) == sh).all()
    names, primary, n_labels = parse_labels(os.path.join(root, "label.csv"))
    # images are stored in random order and split into shards (used for bootstrap later)
    perm = np.random.default_rng(seed).permutation(N)
    shard = (np.arange(N) * shards) // N
    images = np.lib.format.open_memmap(os.path.join(out, "images.npy"), "w+", np.uint8, (N, ih, iw, 3))
    seg = np.lib.format.open_memmap(os.path.join(out, "seg.npy"), "w+", np.uint16, (N, layers, sh, sw))
    pairs, max_layers = [], 0
    jobs = ((pos, idx.iloc[o].to_dict(), root, layers, sh, sw) for pos, o in enumerate(perm))
    with ProcessPoolExecutor(workers) as ex:
        for k, (pos, img, s, il, filled) in enumerate(ex.map(read_one, jobs, chunksize=64)):
            images[pos], seg[pos] = img, s
            pairs += [(pos, l) for l in il]
            max_layers = max(max_layers, filled)
            if k % 5000 == 0:
                print(f"{k}/{N}", flush=True)
    images.flush(); seg.flush()
    imglab = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    img_count = np.zeros(n_labels, dtype=np.int64)
    pix_count = np.zeros(n_labels, dtype=np.int64)
    cat_present = np.zeros((N, len(CATS)), dtype=bool)
    for pos in range(N):
        u, c = np.unique(seg[pos], return_counts=True)
        u, c = u[u > 0], c[u > 0]
        img_count[u] += 1
        pix_count[u] += c
        cat_present[pos, primary[u][primary[u] >= 0]] = True
    for pos, l in imglab:
        img_count[l] += 1
        if primary[l] >= 0:
            cat_present[pos, primary[l]] = True
    np.savez(os.path.join(out, "meta.npz"), perm=perm, shard=shard, label_cat=primary, img_count=img_count,
             pix_count=pix_count, cat_present=cat_present, imglab=imglab,
             split=np.asarray(idx["split"].values[perm].tolist(), dtype="U8"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"names": names, "cats": CATS, "n_labels": n_labels, "N": N, "img_hw": [ih, iw],
                   "seg_hw": [sh, sw], "layers": layers, "max_layers_used": max_layers, "shards": shards,
                   "image_paths": idx["image"].values[perm].tolist()}, f)
    print("done; max label layers used:", max_layers)


def pin(t):
    return t.pin_memory() if torch.cuda.is_available() else t


class Broden:
    """Loader over the preprocessed arrays. A batch never crosses a shard boundary, and a
    background thread prepares the next batch while the GPU works on the current one."""
    image_filter = None   # optional bool mask (N,): only these images are used

    def __init__(self, root, batch_size=32, noise_std=0.0, seed=0, max_images=0):
        info = json.load(open(os.path.join(root, "meta.json")))
        m = np.load(os.path.join(root, "meta.npz"), allow_pickle=True)
        self.images = np.load(os.path.join(root, "images.npy"), mmap_mode="r")
        self.seg = np.load(os.path.join(root, "seg.npy"), mmap_mode="r")
        self.N = self.images.shape[0]
        self.shard, self.label_cat, self.img_count = m["shard"], m["label_cat"], m["img_count"]
        self.pix_count, self.cat_present, self.split = m["pix_count"], m["cat_present"], m["split"]
        self.n_shards = int(self.shard.max()) + 1
        self.n_labels, self.cats, self.names = int(info["n_labels"]), info["cats"], info["names"]
        self.L_used = int(info["max_layers_used"])
        self.imglab = np.zeros((self.N, self.n_labels), dtype=np.uint8)   # scene/texture labels
        self.imglab[m["imglab"][:, 0], m["imglab"][:, 1]] = 1
        if max_images:   # quick check on CPU: first N images (they are stored in random order)
            self.image_filter = np.arange(self.N) < max_images
        self.noise_std, self.seed = noise_std, seed
        self.batches = []
        for s in range(self.n_shards):
            idx = np.nonzero(self.shard == s)[0]
            for b in range(idx.min(), idx.max() + 1, batch_size):
                self.batches.append((s, b, min(b + batch_size, idx.max() + 1)))

    def meta(self):
        valid = (self.img_count >= 10) & (self.label_cat >= 0)   # paper: labels with >= 10 images
        valid[0] = False
        return {"n_labels": self.n_labels, "n_cats": len(self.cats), "names": self.names,
                "label_cat": np.where(self.label_cat >= 0, self.label_cat, 0),
                "label_valid": valid, "cat_names": self.cats}

    def _make(self, s, lo, hi, with_labels, rng):
        rows = slice(lo, hi)
        if self.image_filter is not None:
            r = np.nonzero(self.image_filter[lo:hi])[0] + lo
            if len(r) == 0:
                return None
            if len(r) < hi - lo:
                rows = r
        imgs = torch.from_numpy(np.ascontiguousarray(self.images[rows]))
        if self.noise_std > 0:
            noise = torch.from_numpy(rng.normal(0, self.noise_std * 255, imgs.shape).astype(np.float32))
            imgs = (imgs.float() + noise).clamp_(0, 255).round_().to(torch.uint8)
        if not with_labels:
            return pin(imgs), None
        seg = torch.from_numpy(np.ascontiguousarray(self.seg[rows, : self.L_used]).astype(np.int32))
        il = torch.from_numpy(self.imglab[rows])
        cp = torch.from_numpy(self.cat_present[rows].astype(np.uint8))
        return pin(imgs), (pin(seg), pin(il), pin(cp), s)

    def iter(self, with_labels=False, subsample_batches=None):
        order = list(range(len(self.batches)))
        if subsample_batches:
            order = sorted(np.random.default_rng(1234).choice(order, min(subsample_batches, len(order)), replace=False))
        q = queue.Queue(maxsize=4)
        rng = np.random.default_rng(self.seed)

        def producer():
            for o in order:
                item = self._make(*self.batches[o], with_labels, rng)
                if item is not None:
                    q.put(item)
            q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        while (item := q.get()) is not None:
            yield item


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shards", type=int, default=50)
    a = ap.parse_args()
    preprocess(a.root, a.out, a.workers, shards=a.shards)
