"""Check that every model (especially the converted Caffe ones) works before dissecting it.

imagenette : 1000-way top-1 on Imagenette val for models with ImageNet classes
scenes     : top-1/top-5 scene accuracy on Broden images whose scene label is also a class of the model
  python verify_models.py imagenette caffe-alexnet-imagenet alexnet-imagenet ...
  python verify_models.py scenes caffe-alexnet-places205 alexnet-places365 ...
"""
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as Tr
from torchvision.datasets import ImageFolder

import netdissect as ND
from netdissect.models import load_model, preprocess

mode, names = sys.argv[1], sys.argv[2:]
dev = ND.device(10.0)
W = ND.WEIGHTS
syn = [l.split()[0] for l in open(os.path.join(W, "ilsvrc12/synsets.txt"))]       # ImageNet index order


def class_names(name, n_out):
    """list of class names (Places: 'abbey', 'ice_skating_rink/outdoor'; ImageNet: wnid) for each output."""
    if n_out == 1000:
        return syn
    if n_out == 365:
        return [l.split()[0][3:] for l in open(os.path.join(W, "categories_places365.txt"))]
    if n_out == 205:
        return [l.split()[0][3:] for l in open(os.path.join(W, "categoryIndex_places205.csv"))]
    if n_out == 1183:      # hybrid: 978 ImageNet + 205 Places (names in categoryIndex_hybridCNN.csv)
        return [l.split()[0].split("/", 1)[-1] for l in open(os.path.join(W, "categoryIndex_hybridCNN.csv"))]
    if n_out == 1365:      # hybrid1365: 1000 ImageNet + 365 Places365
        return syn + [l.split()[0][3:] for l in open(os.path.join(W, "categories_places365.txt"))]
    raise ValueError(n_out)


out = {}
with torch.no_grad():
    if mode == "imagenette":
        ds = ImageFolder(os.path.join(ND.IMAGENETTE, "val"), Tr.Compose([Tr.Resize(256), Tr.CenterCrop(224), Tr.ToTensor()]))
        for name in names:
            m = load_model(name, W).to(dev)
            n_out = m.classifier[-1].out_features
            cn = class_names(name, n_out)
            target = torch.tensor([cn.index(ds.classes[t]) if ds.classes[t] in cn else -1 for t in ds.targets])
            correct = n = 0
            for x, idx in torch.utils.data.DataLoader(ds, batch_size=128):
                lg = m(preprocess(m, x.to(dev))).cpu()
                tg = target[n:n + len(x)]
                correct += int((lg.argmax(1) == tg).sum()); n += len(x)
            out[name] = {"outputs": n_out, "imagenette_top1_full_softmax": correct / n,
                         "classes_found": int((target >= 0).sum()) / len(target)}
            print(name, out[name], flush=True)
    else:
        from netdissect.data import Broden
        data = Broden(ND.DATA)
        meta = data.meta()
        scene = [c for c in range(data.n_labels) if meta["label_valid"][c] and meta["cat_names"][meta["label_cat"][c]] == "scene"]
        norm = lambda s: s.replace("/", "-").replace("-s", "").lower()      # 'ice_skating_rink-outdoor-s' style
        for name in names:
            m = load_model(name, W).to(dev)
            cn = [norm(c) for c in class_names(name, m.classifier[-1].out_features)]
            pairs = [(i, cn.index(norm(data.names[c]))) for c in scene for i in np.nonzero(data.imglab[:, c])[0]
                     if norm(data.names[c]) in cn]
            idx = np.array([p[0] for p in pairs]); tgt = torch.tensor([p[1] for p in pairs])
            top1 = top5 = 0
            for s in range(0, len(idx), 256):
                x = torch.from_numpy(np.stack([data.images[i] for i in idx[s:s + 256]])).to(dev).permute(0, 3, 1, 2).float() / 255
                lg = m(preprocess(m, x)).cpu()
                t5 = lg.topk(5, 1).indices
                top1 += int((t5[:, 0] == tgt[s:s + 256]).sum()); top5 += int((t5 == tgt[s:s + 256, None]).any(1).sum())
            out[name] = {"n_images": len(idx), "n_scene_classes_matched": len(set(tgt.tolist())),
                         "top1": top1 / len(idx), "top5": top5 / len(idx)}
            print(name, out[name], flush=True)
p = os.path.join(ND.OUT, f"verify_{mode}.json")
old = json.load(open(p)) if os.path.exists(p) else {}
json.dump({**old, **out}, open(p, "w"), indent=1)
