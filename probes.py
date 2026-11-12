"""Linear probes.

net2vec        : is a concept captured better by a single unit or by a learned direction (linear
                 combination of units)? Directions are learned on Broden train images (pixel-wise
                 weighted BCE) and then dissected exactly like units on Broden val images.
discrimination : paper Sec. 3.6. Linear classifier on global-average-pooled last-conv features for a
                 transfer task (here: Broden scene classification, 30 most frequent scenes) vs. the
                 number of unique object detectors of the model.
  python probes.py net2vec --run alexnet-imagenet --model alexnet-imagenet
  python probes.py discrimination --models alexnet-imagenet=alexnet-imagenet:conv5,...
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

import netdissect as ND
from netdissect import dissect as D
from netdissect.data import Broden
from netdissect.models import FeatureExtractor, arch_of, load_model

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["net2vec", "discrimination"])
ap.add_argument("--run", default="alexnet-imagenet")
ap.add_argument("--model", default="alexnet-imagenet")
ap.add_argument("--layer", default="conv5")
ap.add_argument("--models", default="", help="discrimination: run=model:layer,...")
ap.add_argument("--data", default=ND.DATA)
ap.add_argument("--weights", default=ND.WEIGHTS)
ap.add_argument("--results", default=ND.OUT)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--n-scenes", type=int, default=30)
ap.add_argument("--task", default="scenes", choices=["scenes", "action40"])
ap.add_argument("--action40", default=os.path.join(ND.WORK, "data", "action40"))
a = ap.parse_args()
dev = ND.device(8.0)
torch.manual_seed(0)


def net2vec():
    model = load_model(a.model, a.weights)
    fx = FeatureExtractor(model, arch_of(a.model), [a.layer]).to(dev)
    data = Broden(a.data, batch_size=32)
    meta, names = data.meta(), data.names
    rdir = os.path.join(a.results, a.run)
    tally = pd.read_csv(os.path.join(rdir, f"tally_{a.layer}.csv"))
    T_units = torch.load(os.path.join(rdir, "thresholds.pt"))["T"][a.layer].to(dev)
    # concepts: pixel-level labels of detector units + the 30 most frequent pixel-level labels
    pixel_cats = [meta["cat_names"].index(c) for c in ("object", "part", "material", "color")]
    n2i = {n: i for i, n in enumerate(names) if n}
    cand = [n2i[l] for l in tally[tally.is_detector == 1].label.unique() if meta["label_cat"][n2i[l]] in pixel_cats]
    freq = [c for c in np.argsort(-data.img_count) if meta["label_valid"][c] and meta["label_cat"][c] in pixel_cats]
    concepts = list(dict.fromkeys(cand + freq[:30]))
    C, K = len(concepts), T_units.numel()
    cidx = torch.tensor(concepts, device=dev)
    ccat = torch.tensor(meta["label_cat"][concepts], device=dev)
    start, jump = fx.geometry(a.layer)
    scale = T_units.clamp(min=1e-3)                    # bring units to a similar scale
    lin = torch.nn.Linear(K, C).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    pos_frac = torch.tensor(data.pix_count[concepts] / (data.cat_present[:, meta["label_cat"][concepts]].sum(0) * 112 * 112 + 1),
                            device=dev, dtype=torch.float32).clamp(1e-4, 0.5)
    pos_weight = ((1 - pos_frac) / pos_frac).clamp(max=200.0).view(1, C, 1, 1)
    data.image_filter = data.split == "train"
    grid = None
    for ep in range(a.epochs):
        tot, n = 0.0, 0
        for imgs, (seg, il, cp, s) in data.iter(with_labels=True):
            imgs, seg, cp = imgs.to(dev), seg.to(dev).long(), cp.to(dev).bool()
            with torch.no_grad():
                f = fx(fx.normalize(imgs))[a.layer] / scale.view(1, K, 1, 1)
                if grid is None:
                    grid = D.rf_grid(f.shape[2], f.shape[3], start, jump, 112, 224, dev)
                fu = D.upsample_rf(f, grid)
                tgt = (seg[:, :, None] == cidx.view(1, 1, C, 1, 1)).any(1).float()
                valid = cp[:, ccat].float()            # only images that have the concept's category
            logits = torch.einsum("bkhw,ck->bchw", fu, lin.weight) + lin.bias.view(1, C, 1, 1)
            l = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight, reduction="none")
            loss = (l.mean((2, 3)) * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot, n = tot + loss.item(), n + 1
        print(f"epoch {ep} loss {tot / n:.4f}", flush=True)

    # evaluation on val images: units and directions go through the same dissection
    data.image_filter = data.split == "val"
    W = lin.weight.detach() / scale.view(1, K)
    torch.save({"concepts": concepts, "W": W.cpu(), "b": lin.bias.detach().cpu()}, os.path.join(rdir, "net2vec.pt"))

    def feature_fn(imgs):
        f = fx(fx.normalize(imgs))[a.layer]
        return {"units": f, "dirs": torch.einsum("bkhw,ck->bchw", f, W)}
    T, _ = D.quantile_thresholds(feature_fn, data, ["units", "dirs"], device=dev)
    sums = D.accumulate_iou(feature_fn, data, ["units", "dirs"], T, {"units": (start, jump), "dirs": (start, jump)},
                            meta, data.n_shards, device=dev)
    su, sd = D.summarize(sums["units"], meta), D.summarize(sums["dirs"], meta)
    rows = []
    for j, c in enumerate(concepts):
        bu = int(np.argmax(su["iou"][:, c]))
        rows.append({"concept": names[c], "category": meta["cat_names"][meta["label_cat"][c]],
                     "direction_iou": float(sd["iou"][j, c]), "best_unit": bu, "best_unit_iou": float(su["iou"][bu, c]),
                     "has_detector_unit": bool(((su["best"] == c) & su["is_detector"]).any())})
    df = pd.DataFrame(rows).sort_values("direction_iou", ascending=False)
    df["gain"] = df.direction_iou - df.best_unit_iou
    df.to_csv(os.path.join(rdir, f"net2vec_{a.layer}.csv"), index=False)
    summ = {"n_concepts": C, "mean_direction_iou": float(df.direction_iou.mean()),
            "mean_best_unit_iou": float(df.best_unit_iou.mean()), "frac_direction_better": float((df.gain > 0).mean()),
            "wilcoxon_p": float(stats.wilcoxon(df.direction_iou, df.best_unit_iou).pvalue),
            "concepts_without_unit_but_direction_iou_gt_0.04":
                df[(~df.has_detector_unit) & (df.direction_iou > 0.04)].concept.tolist()}
    json.dump(summ, open(os.path.join(rdir, f"net2vec_{a.layer}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


def linear_svm(Xtr, ytr, Xte, yte, n_cls, C=0.001):
    """one-vs-rest linear SVM, squared hinge, L2 (liblinear's default form), on L2-normalised features.
    Returns mean per-class accuracy (as reported in the paper)."""
    nrm = lambda X: X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Xtr, Xte = nrm(Xtr).to(dev), nrm(Xte).to(dev)
    Y = torch.full((len(ytr), n_cls), -1.0, device=dev)
    Y[torch.arange(len(ytr)), ytr] = 1.0
    W = torch.zeros(Xtr.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=1000, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.5 * (W ** 2).sum() + C * (torch.clamp(1 - Y * (Xtr @ W + b), min=0) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    pred = (Xte @ W + b).argmax(1).cpu()
    return float(np.mean([(pred[yte == c] == c).float().mean().item() for c in range(n_cls) if (yte == c).any()]))


def action40_images():
    """Stanford40: 224x224 (warped) uint8 images, labels, train mask (official split)."""
    from PIL import Image
    d = a.action40
    acts = [l.split()[0] for l in open(os.path.join(d, "ImageSplits", "actions.txt")).read().split("\n")[1:] if l.strip()]
    X, y, tr = [], [], []
    for split in ("train", "test"):
        for fn in open(os.path.join(d, "ImageSplits", f"{split}.txt")).read().split():
            X.append(np.asarray(Image.open(os.path.join(d, "JPEGImages", fn)).convert("RGB").resize((224, 224), Image.BILINEAR)))
            y.append(acts.index(fn.rsplit("_", 1)[0])); tr.append(split == "train")
    return np.stack(X), torch.tensor(y), np.array(tr), len(acts)


def discrimination_action40():
    X_img, y, tr, n_cls = action40_images()
    rows = []
    for spec in a.models.split(","):
        run, rest = spec.split("=")
        name, layer = rest.rsplit(":", 1)
        if not os.path.exists(os.path.join(a.results, run, "summary.json")):
            print("skipped (not dissected):", run, flush=True)
            continue
        fx = FeatureExtractor(load_model(name, a.weights), arch_of(name), [layer]).to(dev)
        with torch.no_grad():
            X = torch.cat([fx(fx.normalize(torch.from_numpy(X_img[s:s + 128]).to(dev)))[layer].mean((2, 3)).cpu()
                           for s in range(0, len(X_img), 128)])
        acc = linear_svm(X[tr], y[tr], X[~tr], y[~tr], n_cls)
        summ = json.load(open(os.path.join(a.results, run, "summary.json")))["variants"][layer]
        rows.append({"run": run, "layer": layer, "action40_acc": acc,
                     "unique_object_detectors": summ["unique_per_cat"]["object"], "unique_total": summ["unique_total"]})
        print(rows[-1], flush=True)
        json.dump({"task": f"Stanford40 actions, {int(tr.sum())} train / {int((~tr).sum())} test, linear SVM C=0.001 on GAP last-conv",
                   "rows": rows}, open(os.path.join(a.results, "discrimination_action40.json"), "w"), indent=1)   # saved after every model


def discrimination():
    data = Broden(a.data)
    meta = data.meta()
    scene = meta["cat_names"].index("scene")
    scenes = [c for c in range(data.n_labels) if meta["label_valid"][c] and meta["label_cat"][c] == scene]
    top = [scenes[i] for i in np.argsort(-data.imglab[:, scenes].sum(0))[: a.n_scenes]]
    has = data.imglab[:, top]
    idx = np.nonzero(has.sum(1) == 1)[0]            # images with exactly one of these scenes
    y = torch.from_numpy(has[idx].argmax(1)).long()
    tr = data.split[idx] == "train"
    rows = []
    for spec in a.models.split(","):
        run, rest = spec.split("=")
        name, layer = rest.rsplit(":", 1)
        if not os.path.exists(os.path.join(a.results, run, "summary.json")):
            print("skipped (not dissected):", run, flush=True)
            continue
        fx = FeatureExtractor(load_model(name, a.weights), arch_of(name), [layer]).to(dev)
        with torch.no_grad():
            X = torch.cat([fx(fx.normalize(torch.from_numpy(np.stack([data.images[i] for i in idx[s:s + 128]])).to(dev)))
                           [layer].mean((2, 3)).cpu() for s in range(0, len(idx), 128)])
        # multinomial logistic regression (L2, LBFGS), mean per-class accuracy on val
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xva = ((X[tr] - mu) / sd).to(dev), ((X[~tr] - mu) / sd).to(dev)
        ytr, yva = y[tr].to(dev), y[~tr].to(dev)
        W = torch.zeros(X.shape[1], len(top), device=dev, requires_grad=True)
        b = torch.zeros(len(top), device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-3 * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        pred = (Xva @ W + b).argmax(1)
        acc = float(np.mean([(pred[yva == c] == c).float().mean().item() for c in range(len(top)) if (yva == c).any()]))
        summ = json.load(open(os.path.join(a.results, run, "summary.json")))["variants"][layer]
        rows.append({"run": run, "layer": layer, "scene_probe_acc": acc,
                     "unique_object_detectors": summ["unique_per_cat"]["object"], "unique_total": summ["unique_total"]})
        print(rows[-1], flush=True)
        json.dump({"task": f"Broden top-{a.n_scenes} scenes, n_train={int(tr.sum())}, n_val={int((~tr).sum())}",
                   "rows": rows}, open(os.path.join(a.results, "discrimination.json"), "w"), indent=1)


if a.mode == "net2vec":
    net2vec()
elif a.task == "action40":
    discrimination_action40()
else:
    discrimination()
