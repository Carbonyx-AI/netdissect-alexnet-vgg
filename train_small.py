"""Train AlexNet variants from scratch on Imagenette for the training-condition experiments of the
paper (Sec. 3.5: iterations, initialisation, dropout, batch norm; Sec. 3.7: width). This is a
reduced-scale version: the paper used Places205 (2.4M images), we use 9,469 images.
  python train_small.py --variant baseline     (baseline | repeat1 | repeat2 | nodropout | batchnorm | gapwide)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import netdissect as ND
from netdissect.models import MEAN, STD, build_small

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--imagenette", default=ND.IMAGENETTE)
ap.add_argument("--out", default=ND.TRAINED)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=None, help="default: 0 (repeat1/2 of the first study use 1/2)")
a = ap.parse_args()


def load_split(split, size=256):
    """resize short side to 256, centre crop 256x256, keep as uint8."""
    classes = sorted(os.listdir(os.path.join(a.imagenette, split)))
    X, y = [], []
    for ci, c in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(a.imagenette, split, c))):
            im = Image.open(os.path.join(a.imagenette, split, c, fn)).convert("RGB")
            w, h = im.size
            s = size / min(w, h)
            im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
            l, t = (im.size[0] - size) // 2, (im.size[1] - size) // 2
            X.append(np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8))
            y.append(ci)
    return np.stack(X), np.array(y)


dev = ND.device(8.0)
seed = a.seed if a.seed is not None else {"baseline": 0, "repeat1": 1, "repeat2": 2}.get(a.variant, 0)
arch = {"repeat1": "baseline", "repeat2": "baseline"}.get(a.variant, a.variant)
run = a.variant if a.seed is None else f"{a.variant}-s{a.seed}"
torch.manual_seed(seed); np.random.seed(seed)
os.makedirs(a.out, exist_ok=True)
cache = os.path.join(a.out, "imagenette_cache.npz")
if not os.path.exists(cache):
    (Xtr, ytr), (Xva, yva) = load_split("train"), load_split("val")
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
d = np.load(cache)
Xtr, ytr, Xva, yva = (torch.from_numpy(d[k]) for k in ("Xtr", "ytr", "Xva", "yva"))
mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)


def prep(xb, train):
    x = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    if train:                      # random 224 crop + horizontal flip
        i, j = np.random.randint(0, 33, 2)
        x = x[:, :, i:i + 224, j:j + 224]
        if np.random.rand() < 0.5:
            x = x.flip(3)
    else:
        x = x[:, :, 16:240, 16:240]
    return (x - mean) / std


model = build_small(arch).to(dev)
for mod in model.modules():
    if isinstance(mod, nn.ReLU):
        mod.inplace = False
total_steps = a.epochs * int(np.ceil(len(Xtr) / a.batch))
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.15)
vdir = os.path.join(a.out, run)
os.makedirs(vdir, exist_ok=True)
snap = {0, 50, 100, 200, 400, 800, 1600, total_steps} if arch == "baseline" else {total_steps}


def evaluate():
    model.eval()
    with torch.no_grad():
        correct = sum((model(prep(Xva[s:s + 256], False)).argmax(1).cpu() == yva[s:s + 256]).sum().item()
                      for s in range(0, len(Xva), 256))
    model.train()
    return correct / len(Xva)


log, step, t0 = [], 0, time.time()
if 0 in snap:
    torch.save(model.state_dict(), os.path.join(vdir, "iter0.pt"))
model.train()
for ep in range(a.epochs):
    perm = torch.randperm(len(Xtr))
    for s in range(0, len(Xtr), a.batch):
        idx = perm[s:s + a.batch]
        loss = F.cross_entropy(model(prep(Xtr[idx], True)), ytr[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        step += 1
        if step in snap:
            torch.save(model.state_dict(), os.path.join(vdir, f"iter{step}.pt"))
    if ep % 10 == 9 or ep == a.epochs - 1:
        log.append({"epoch": ep + 1, "step": step, "loss": loss.item(), "val_acc": evaluate(), "time_s": time.time() - t0})
        print(log[-1], flush=True)
json.dump({"variant": a.variant, "seed": seed, "epochs": a.epochs, "steps": total_steps, "lr": a.lr,
           "final_val_acc": log[-1]["val_acc"], "log": log, "snapshots": sorted(snap)},
          open(os.path.join(vdir, "train_log.json"), "w"), indent=1)
