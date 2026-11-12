"""Dissect some layers of one model.

  python run_dissect.py --model alexnet-imagenet --layers conv1,conv2,conv3,conv4,conv5
  python run_dissect.py --model alexnet-places365 --layers conv5 --rotate 0,0.2,0.4,0.6,0.8,1   # Sec 3.2
  python run_dissect.py --model alexnet-imagenet --layers conv5 --negative          # bottom 0.5%, pre-ReLU
  python run_dissect.py --model alexnet-imagenet --layers conv5 --quantile 0.01      # threshold check
  python run_dissect.py --model alexnet-imagenet --layers conv5 --noise 0.1          # noisy images
  python run_dissect.py --model alexnet-imagenet --layers conv5 --randomize-from conv3   # sanity check
Writes results/<name>/: thresholds.pt, sums_<layer>.npz, tally_<layer>.csv, summary.json, log.txt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import netdissect as ND
from netdissect import dissect as D
from netdissect.data import Broden
from netdissect.models import FeatureExtractor, arch_of, conv_layers, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--layers", required=True)
ap.add_argument("--name", default=None)
ap.add_argument("--data", default=ND.DATA)
ap.add_argument("--weights", default=ND.WEIGHTS)
ap.add_argument("--out", default=ND.OUT)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--quantile", type=float, default=0.005)
ap.add_argument("--iou", type=float, default=0.04)
ap.add_argument("--negative", action="store_true")
ap.add_argument("--rotate", default=None, help="comma list of alpha values (one layer only)")
ap.add_argument("--rot-seed", type=int, default=0)
ap.add_argument("--noise", type=float, default=0.0, help="std of Gaussian pixel noise, fraction of 255")
ap.add_argument("--randomize-from", default=None, help="fc or a conv layer: re-init it and everything above")
ap.add_argument("--max-images", type=int, default=0, help="quick check with only N images")
ap.add_argument("--gpu-mem-gb", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = ND.device(a.gpu_mem_gb)
torch.manual_seed(a.seed)
arch, layers = arch_of(a.model), a.layers.split(",")
outdir = os.path.join(a.out, a.name or a.model)
os.makedirs(outdir, exist_ok=True)
logf = open(os.path.join(outdir, "log.txt"), "a")


def log(*x):
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    logf.write(s + "\n"); logf.flush()


log(f"=== {time.ctime()} {vars(a)}")
model = load_model(a.model, a.weights, seed=a.seed)
if a.randomize_from:   # cascading randomisation: re-init classifier and all conv layers from this one up
    g = torch.Generator().manual_seed(a.seed + 99)
    convs = conv_layers(model, arch)
    names = [l.name for l in convs]
    k0 = len(names) if a.randomize_from == "fc" else names.index(a.randomize_from)
    for mod in model.classifier.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0, 0.01, generator=g); nn.init.zeros_(mod.bias)
    for l in convs[k0:]:
        nn.init.kaiming_normal_(model.features[l.conv_idx].weight, mode="fan_out", nonlinearity="relu", generator=g)
        nn.init.zeros_(model.features[l.conv_idx].bias)
    log(f"randomized classifier + layers {names[k0:]}")

fx = FeatureExtractor(model, arch, layers, pre_relu=a.negative).to(dev)
sign = -1.0 if a.negative else 1.0
data = Broden(a.data, batch_size=a.batch, noise_std=a.noise, seed=a.seed, max_images=a.max_images)
meta = data.meta()
log(f"Broden: N={data.N}, labels={meta['label_valid'].sum()} valid, shards={data.n_shards}")

rot = {}
if a.rotate:
    K = next(l.channels for l in conv_layers(model, arch) if l.name == layers[0])
    Q = D.random_rotation(K, seed=a.rot_seed)
    np.save(os.path.join(outdir, "Q.npy"), Q)
    for al in [float(x) for x in a.rotate.split(",")]:
        Qa = D.fractional_rotation(Q, al)
        log(f"rotation alpha={al}: orthogonality error {np.abs(Qa @ Qa.T - np.eye(K)).max():.1e}, "
            f"det {np.linalg.det(Qa):.6f}")
        rot[f"{layers[0]}@rot{al:g}"] = torch.tensor(Qa, dtype=torch.float32, device=dev)


def feature_fn(imgs_u8):
    feats = fx(fx.normalize(imgs_u8))
    if rot:
        f = feats.pop(layers[0])
        for k, Qa in rot.items():
            feats[k] = torch.einsum("ij,bjhw->bihw", Qa, f)
    return feats


variants = list(rot) if rot else layers
geometry = {v: fx.geometry(v.split("@")[0]) for v in variants}
t0 = time.time()
T, totals = D.quantile_thresholds(feature_fn, data, variants, q_top=a.quantile, sign=sign, device=dev, log=log)
torch.save({"T": T, "totals": totals, "sign": sign, "q": a.quantile}, os.path.join(outdir, "thresholds.pt"))
t1 = time.time()
sums = D.accumulate_iou(feature_fn, data, variants, T, geometry, meta, data.n_shards, sign=sign, device=dev, log=log)
summary = {"args": vars(a), "runtime_s": {"quantile": t1 - t0, "iou": time.time() - t1}, "variants": {}}
for v in variants:
    np.savez_compressed(os.path.join(outdir, f"sums_{v}.npz"), **sums[v])
    s = D.summarize(sums[v], meta, iou_thresh=a.iou)
    u = summary["variants"][v] = D.unique_detectors(s, meta)
    log(f"{v}: unique={u['unique_total']} {u['unique_per_cat']} detector_units={u['detector_units']}/{len(s['best'])}")
    with open(os.path.join(outdir, f"tally_{v}.csv"), "w") as f:
        f.write("unit,label,category,iou,is_detector\n")
        for k in np.argsort(-s["best_iou"]):
            c = s["best"][k]
            f.write(f"{k},{meta['names'][c]},{meta['cat_names'][meta['label_cat'][c]]},"
                    f"{s['best_iou'][k]:.4f},{int(s['is_detector'][k])}\n")
json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
log(f"=== done in {time.time() - t0:.0f}s")
