"""Do the labelled units matter for the prediction? (+ accuracy check of the rotation experiment)

Permutation importance (Molnar) with conv units as features: shuffle a unit's activation map
across Imagenette val images and see how much the error goes up. Also shuffle all units with the
same concept label together and compare with random groups of the same size.
  python run_fidelity.py --model alexnet-imagenet --layer conv5
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as Tr
from scipy import stats
from torchvision.datasets import ImageFolder

import netdissect as ND
from netdissect.dissect import random_rotation
from netdissect.models import MEAN, STD, arch_of, get_layer, load_model, preprocess

# Imagenette folder name -> (ImageNet class index, name)
CLASSES = {"n01440764": (0, "tench"), "n02102040": (217, "English springer"), "n02979186": (482, "cassette player"),
           "n03000684": (491, "chain saw"), "n03028079": (497, "church"), "n03394916": (566, "French horn"),
           "n03417042": (569, "garbage truck"), "n03425413": (571, "gas pump"), "n03445777": (574, "golf ball"),
           "n03888257": (701, "parachute")}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="alexnet-imagenet")
ap.add_argument("--layer", default="conv5")
ap.add_argument("--imagenette", default=ND.IMAGENETTE)
ap.add_argument("--results", default=ND.OUT)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--random-groups", type=int, default=1000)
ap.add_argument("--prereg-null", type=int, default=10000)
ap.add_argument("--gpu-mem-gb", type=float, default=10.0)
a = ap.parse_args()
torch.set_grad_enabled(False)
dev = ND.device(a.gpu_mem_gb)
torch.manual_seed(0)
rng = np.random.default_rng(0)
model = load_model(a.model).to(dev)
li = get_layer(model, arch_of(a.model), a.layer)
rdir = os.path.join(a.results, a.model)

caffe = hasattr(model, "preprocess")     # Caffe models get their own input format (BGR, 0-255, mean)
tf = Tr.Compose([Tr.Resize(256), Tr.CenterCrop(224), Tr.ToTensor()] + ([] if caffe else [Tr.Normalize(MEAN, STD)]))
ds = ImageFolder(os.path.join(a.imagenette, "val"), transform=tf)
names = [CLASSES[c][1] for c in ds.classes]
cls_idx = torch.tensor([CLASSES[c][0] for c in ds.classes], device=dev)
y = torch.tensor(ds.targets, device=dev)
feats = []
for x, _ in torch.utils.data.DataLoader(ds, batch_size=64, num_workers=6 if sys.platform == "linux" else 0):
    h = preprocess(model, x.to(dev)) if caffe else x.to(dev)
    for mod in model.features[: li.relu_idx + 1]:
        h = mod(h)
    feats.append(h)
Fm = torch.cat(feats)                 # (N,K,h,w) activations of the dissected layer
N, K = Fm.shape[:2]


def head(f, bs=512):                  # rest of the network after the dissected layer
    outs = []
    for s in range(0, f.shape[0], bs):
        h = f[s:s + bs]
        for mod in model.features[li.relu_idx + 1:]:
            h = mod(h)
        outs.append(model.classifier(torch.flatten(model.avgpool(h), 1)))
    return torch.cat(outs)


def scores(logits):
    sub = logits[:, cls_idx]                                  # 10-way
    acc10 = (sub.argmax(1) == y).float()
    acc1000 = (logits.argmax(1) == cls_idx[y]).float()        # 1000-way
    per = lambda acc: torch.stack([acc[y == c].mean() for c in range(10)]).cpu().numpy()
    return acc10.mean().item(), acc1000.mean().item(), F.cross_entropy(sub, y, reduction="none").mean().item(), per(acc10), per(acc1000)


base = scores(head(Fm))
out = {"model": a.model, "layer": a.layer, "N": N, "K": K,
       "base": {"acc10": base[0], "acc1000": base[1], "loss10": base[2], "per_class": dict(zip(names, base[3].tolist())),
                "per_class1000": dict(zip(names, base[4].tolist()))}}
print("baseline", out["base"])

# rotation check: g(Q^T Q f) must equal g(f); g(Q f) without the inverse should break the network
Q = torch.tensor(random_rotation(K, seed=0), dtype=torch.float32, device=dev)
r = torch.einsum("ij,bjhw->bihw", Q, Fm)
lg0, lg1 = head(Fm), head(torch.einsum("ji,bjhw->bihw", Q, r))
out["rotation_check"] = {"acc10_rotated_with_inverse": scores(lg1)[0], "acc1000_rotated_with_inverse": scores(lg1)[1],
                         "max_abs_logit_diff": (lg0 - lg1).abs().max().item(),
                         "top1_agreement": (lg0.argmax(1) == lg1.argmax(1)).float().mean().item(),
                         "acc10_rotated_without_inverse": scores(head(r))[0]}
print("rotation check", out["rotation_check"])
del r


def shuffle(units):
    perm = torch.randperm(N, device=dev)
    Fp = Fm.clone()
    idx = torch.tensor(units, device=dev)
    Fp[:, idx] = Fm[perm][:, idx]
    return scores(head(Fp))


# single-unit permutation importance
imp_acc, imp_loss = np.zeros((K, a.repeats)), np.zeros((K, a.repeats))
for k in range(K):
    for t in range(a.repeats):
        s = shuffle([k])
        imp_acc[k, t], imp_loss[k, t] = base[0] - s[0], s[2] / base[2]     # loss ratio as in the book
units = pd.DataFrame({"unit": np.arange(K), "imp_acc": imp_acc.mean(1), "imp_acc_sd": imp_acc.std(1),
                      "imp_loss_ratio": imp_loss.mean(1)})
units = units.merge(pd.read_csv(os.path.join(rdir, f"tally_{a.layer}.csv")), on="unit")
units.to_csv(os.path.join(rdir, f"fidelity_units_{a.layer}.csv"), index=False)
det = units.is_detector == 1
sp = stats.spearmanr(units.iou, units.imp_loss_ratio)
out["single_unit"] = {"n_detectors": int(det.sum()),
                      "mean_loss_ratio_detectors": float(units.imp_loss_ratio[det].mean()),
                      "mean_loss_ratio_nondetectors": float(units.imp_loss_ratio[~det].mean()),
                      "mannwhitney_p": float(stats.mannwhitneyu(units.imp_loss_ratio[det], units.imp_loss_ratio[~det]).pvalue),
                      "spearman_iou_vs_importance": float(sp.statistic), "spearman_p": float(sp.pvalue),
                      "max_single_unit_acc_drop": float(units.imp_acc.max())}
print("single unit", out["single_unit"])

# grouped permutation: all detector units with the same label vs random groups of same size
groups = units[det].groupby("label").unit.apply(list)
groups = groups[groups.apply(len) >= 2]


def group_drop(u):
    s = shuffle(u)
    return np.stack([base[3] - s[3], base[4] - s[4]])        # (2,10) per-class drop, 10-way and 1000-way


rows, null_cache = [], {}
for label, us in groups.items():
    d = np.mean([group_drop(us) for _ in range(a.repeats)], 0)
    n = len(us)
    if n not in null_cache:
        null_cache[n] = np.array([group_drop(list(rng.choice(K, n, replace=False))) for _ in range(a.random_groups)])
    null = null_cache[n]
    for c in range(10):
        rows.append({"concept": label, "n_units": n, "class": names[c], "acc_drop": d[0, c],
                     "null_mean": null[:, 0, c].mean(), "null_95": np.quantile(null[:, 0, c], 0.95),
                     "p_value": (1 + np.sum(null[:, 0, c] >= d[0, c])) / (1 + len(null)),
                     "acc1000_drop": d[1, c], "null1000_mean": null[:, 1, c].mean(),
                     "null1000_95": np.quantile(null[:, 1, c], 0.95),
                     "p_value_1000": (1 + np.sum(null[:, 1, c] >= d[1, c])) / (1 + len(null))})
# the one test fixed before seeing results: 'dog' units -> dog class (English springer)
if "dog" in groups.index:
    dog, c = groups["dog"], names.index("English springer")
    d = np.mean([group_drop(dog) for _ in range(a.repeats)], 0)[:, c]
    null = np.array([group_drop(list(rng.choice(K, len(dog), replace=False)))[:, c] for _ in range(a.prereg_null)])
    out["prespecified_dog_springer"] = {
        "n_dog_units": len(dog), "acc10_drop": float(d[0]), "acc1000_drop": float(d[1]),
        "null1000_mean": float(null[:, 1].mean()), "null1000_sd": float(null[:, 1].std()),
        "null1000_max": float(null[:, 1].max()), "n_null": len(null),
        "p_value_1000": float((1 + np.sum(null[:, 1] >= d[1])) / (1 + len(null))),
        "z_1000": float((d[1] - null[:, 1].mean()) / null[:, 1].std())}
    print("pre-specified test", out["prespecified_dog_springer"])
g = pd.DataFrame(rows, columns=["concept", "n_units", "class", "acc_drop", "null_mean", "null_95", "p_value", "acc1000_drop",
                                "null1000_mean", "null1000_95", "p_value_1000"])
g.to_csv(os.path.join(rdir, f"fidelity_groups_{a.layer}.csv"), index=False)
out["grouped"] = {"n_groups": int(len(groups)), "n_tests": len(g),
                  "significant_bonferroni": g[g.p_value_1000 * len(g) < 0.05].sort_values("acc1000_drop", ascending=False).head(30).to_dict("records"),
                  "top_concept_per_class": g.sort_values("acc1000_drop", ascending=False).groupby("class").head(1).to_dict("records")}
json.dump(out, open(os.path.join(rdir, f"fidelity_{a.layer}.json"), "w"), indent=1, default=float)
