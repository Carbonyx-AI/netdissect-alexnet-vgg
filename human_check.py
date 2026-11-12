"""Human evaluation of unit labels (paper Sec. 3.1, Table 3) as a ~10 minute job.

make  : picks detector units (default: 8 per layer of AlexNet-Places205 conv1-conv5, the paper's
        model, + 10 of VGG16-Places365 conv5_3), draws each unit's 10 most activating Broden images with
        the unit mask, and pairs it with a phrase: the Network Dissection label for a random half,
        a decoy label of the same category for the other half. Writes one offline HTML page
        (Y/N keys, autosave, download) and a separate answer key.
score : reads one or two answer files and writes results/human_check.json + a figure:
        per layer, % of dissection labels judged descriptive (paper Table 3 "Network Dissection" row),
        % of decoys judged descriptive (control), and agreement between two raters if given.
  python human_check.py make
  python human_check.py score answers_A.json [answers_B.json]
"""
import argparse
import base64
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

import netdissect as ND
from netdissect import dissect as D
from netdissect.data import Broden
from netdissect.models import FeatureExtractor, arch_of, load_model

PAPER = {"conv1": (37, 82), "conv2": (56, 76), "conv3": (54, 83), "conv4": (59, 82), "conv5": (71, 91)}   # Table 3
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["make", "score"])
ap.add_argument("answers", nargs="*")
ap.add_argument("--plan", default="caffe-alexnet-places205:conv1,conv2,conv3,conv4,conv5:8;caffe-vgg16-places365:conv5_3:10")
ap.add_argument("--data", default=ND.DATA)
ap.add_argument("--weights", default=ND.WEIGHTS)
ap.add_argument("--results", default=ND.OUT)
a = ap.parse_args()
DIR = os.path.join(a.results, "human_check")


def panel(data, fx, layer, T, u, top, start, jump, dev):
    """10 thumbnails (2 x 5) with everything outside the unit mask darkened."""
    imgs = torch.from_numpy(np.stack([data.images[i] for i in top])).to(dev)
    with torch.no_grad():
        f = fx(fx.normalize(imgs))[layer][:, u:u + 1]
        g = D.rf_grid(f.shape[2], f.shape[3], start, jump, data.seg.shape[-1], imgs.shape[1], dev)
        M = (D.upsample_rf(f, g)[:, 0] > T[u]).cpu().numpy()
    tiles = []
    for j, i in enumerate(top):
        img = np.asarray(data.images[i]).astype(np.float32)
        m = np.asarray(Image.fromarray(M[j].astype(np.uint8) * 255).resize((224, 224), Image.NEAREST)) > 0
        img[~m] *= 0.25
        tiles.append(np.asarray(Image.fromarray(img.astype(np.uint8)).resize((150, 150))))
    rows = [np.concatenate(tiles[r * 5:(r + 1) * 5], 1) for r in range(2)]
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(rows, 0)).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def make():
    dev = ND.device(8.0)
    rng = np.random.default_rng(2017)
    data = Broden(a.data, batch_size=64)
    meta = data.meta()
    items, key = [], []
    for spec in a.plan.split(";"):
        model, layers, n = spec.split(":")
        fx = FeatureExtractor(load_model(model, a.weights), arch_of(model), layers.split(",")).to(dev)
        rdir = os.path.join(a.results, model)
        T_all = torch.load(os.path.join(rdir, "thresholds.pt"))["T"]
        with torch.no_grad():       # max activation of every unit on every image (one pass for all layers)
            mx = {l: [] for l in layers.split(",")}
            for imgs, _ in data.iter():
                feats = fx(fx.normalize(imgs.to(dev)))
                for l in mx:
                    mx[l].append(feats[l].amax((2, 3)).cpu())
        for l in layers.split(","):
            tally = pd.read_csv(os.path.join(rdir, f"tally_{l}.csv")).set_index("unit")
            det = tally[tally.is_detector == 1]
            order = np.argsort(-torch.cat(mx[l]).numpy(), 0)
            start, jump = fx.geometry(l)
            T = T_all[l].to(dev)
            for u in rng.choice(det.index.values, size=min(int(n), len(det)), replace=False):
                true, cat = det.loc[u, "label"], det.loc[u, "category"]
                pool = sorted(set(tally[(tally.category == cat) & (tally.label != true)].label.dropna())) or \
                    [meta["names"][c] for c in np.nonzero(meta["label_valid"] & (meta["label_cat"] == meta["cat_names"].index(cat)))[0]
                     if meta["names"][c] != true]
                use_true = bool(rng.random() < 0.5)
                shown = true if use_true else str(rng.choice(pool))
                key.append({"model": model, "layer": l, "unit": int(u), "true_label": true, "category": cat, "shown": shown,
                            "is_true_label": use_true, "iou": float(det.loc[u, "iou"])})
                items.append({"img": panel(data, fx, l, T, int(u), order[:10, u], start, jump, dev),
                              "phrase": shown.replace("-c", " (colour)").replace("-s", " (scene)")})
    perm = rng.permutation(len(items))                          # mix layers and models
    items = [dict(items[i], id=k) for k, i in enumerate(perm)]
    key = [dict(key[i], id=k) for k, i in enumerate(perm)]
    os.makedirs(DIR, exist_ok=True)
    json.dump(key, open(os.path.join(DIR, "KEY_do_not_open.json"), "w"), indent=1)
    open(os.path.join(DIR, "human_check.html"), "w").write(HTML.replace("__ITEMS__", json.dumps(items)))
    print(len(items), "items written to", DIR)


def score():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    key = pd.DataFrame(json.load(open(os.path.join(DIR, "KEY_do_not_open.json"))))
    raters = [json.load(open(p)) for p in a.answers]
    for r in raters:
        key[r["rater"]] = key.id.map(lambda i: r["answers"].get(str(i)))
    names = [r["rater"] for r in raters]
    rows = []
    for (m, l), g in key.groupby(["model", "layer"], sort=False):
        row = {"model": m, "layer": l, "n_units": len(g)}
        for n in names:
            yes = g[n] == "yes"
            row[f"{n}_true_label_accepted"] = float(yes[g.is_true_label].mean()) if g.is_true_label.any() else None
            row[f"{n}_decoy_accepted"] = float(yes[~g.is_true_label].mean()) if (~g.is_true_label).any() else None
        if m == "caffe-alexnet-places205" and l in PAPER:
            row["paper_network_dissection"], row["paper_human_consistency"] = PAPER[l]
        rows.append(row)
    out = {"per_layer": rows}
    for n in names:
        ans = key[n].dropna()
        k = key.loc[ans.index]
        yes, truth = (ans == "yes").values, k.is_true_label.values
        po = float(np.mean(yes == truth))                          # accuracy of telling true labels from decoys
        pe = yes.mean() * truth.mean() + (1 - yes.mean()) * (1 - truth.mean())
        out[n] = {"answered": int(len(ans)), "true_label_accepted": float(yes[truth].mean()),
                  "decoy_accepted": float(yes[~truth].mean()), "agreement_with_method": po,
                  "cohen_kappa_with_method": float((po - pe) / (1 - pe))}
    if len(names) == 2:
        both = key[names].dropna()
        a1, a2 = both[names[0]] == "yes", both[names[1]] == "yes"
        po = float((a1 == a2).mean()); pe = a1.mean() * a2.mean() + (1 - a1.mean()) * (1 - a2.mean())
        out["between_raters"] = {"n": int(len(both)), "agreement": po, "cohen_kappa": float((po - pe) / (1 - pe))}
    json.dump(out, open(os.path.join(a.results, "human_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    df = pd.DataFrame(rows)
    al = df[df.model == "caffe-alexnet-places205"]
    if len(al) and names:
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(len(al))
        ax.bar(x - 0.27, al[f"{names[0]}_true_label_accepted"] * 100, 0.27, label="ours: true labels accepted", color="#2a78d6")
        ax.bar(x, al[f"{names[0]}_decoy_accepted"] * 100, 0.27, label="ours: decoys accepted", color="#eb6834")
        ax.bar(x + 0.27, al.paper_network_dissection, 0.27, label="paper (Table 3)", color="#9a9a9a")
        ax.set_xticks(x, al.layer); ax.set_ylabel("% judged descriptive"); ax.set_ylim(0, 100)
        ax.set_title("Human check, AlexNet-Places205", fontsize=10, loc="left")
        ax.legend(fontsize=7, frameon=False)
        os.makedirs(os.path.join(a.results, "figures"), exist_ok=True)
        fig.savefig(os.path.join(a.results, "figures", "fig_human_check.png"), dpi=200, bbox_inches="tight")


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Unit Label Check</title><style>
body{font:16px sans-serif;margin:0;background:#f6f6f4;color:#111} #w{max-width:800px;margin:auto;padding:14px}
img{width:100%;border-radius:6px} .p{font-size:22px;margin:10px 0} .k{display:inline-block;padding:10px 26px;margin:6px;border-radius:8px;border:1px solid #999;cursor:pointer;background:#fff}
#bar{height:6px;background:#2a78d6;width:0} .m{color:#666;font-size:14px}</style></head><body><div id="bar"></div><div id="w">
<div id="intro"><h2>Unit label check (about 10 minutes)</h2><p>Each screen shows the 10 images that activate one hidden unit of a CNN most.
Bright regions are where the unit fires; the rest is darkened. Press <b>Y</b> if the phrase describes <b>most</b> of the bright regions,
otherwise <b>N</b>. Do not think too long. Your answers are saved in this browser; at the end press Download.</p>
Name: <input id="r"> <span class="k" onclick="go()">Start</span>
<p class="m">Second person on the same computer? <a href="#" onclick="localStorage.removeItem('uc');location.reload()">start fresh</a></p></div>
<div id="q" style="display:none"><div class="m" id="c"></div><div class="p">Does <b id="ph"></b> describe most bright regions?</div>
<img id="im"><div><span class="k" onclick="ans('yes')">Y = yes</span><span class="k" onclick="ans('no')">N = no</span>
<span class="k" onclick="back()">&larr; back</span></div></div>
<div id="end" style="display:none"><h2>Done, thank you.</h2><span class="k" onclick="dl()">Download answers</span></div></div>
<script>const I=__ITEMS__;let S=JSON.parse(localStorage.getItem('uc')||'{"a":{},"i":0,"r":""}');
function save(){localStorage.setItem('uc',JSON.stringify(S))}
function go(){S.r=document.getElementById('r').value||S.r||'rater';save();intro.style.display='none';show()}
function show(){if(S.i>=I.length){q.style.display='none';end.style.display='block';return}q.style.display='block';const t=I[S.i];
ph.textContent='"'+t.phrase+'"';im.src='data:image/jpeg;base64,'+t.img;c.textContent=(S.i+1)+' / '+I.length;bar.style.width=(100*S.i/I.length)+'%'}
function ans(v){S.a[I[S.i].id]=v;S.i++;save();show()} function back(){if(S.i>0){S.i--;save();show()}}
document.onkeydown=e=>{if(q.style.display!='block')return;if(e.key=='y'||e.key=='Y')ans('yes');if(e.key=='n'||e.key=='N')ans('no');if(e.key=='Backspace')back()};
function dl(){const l=document.createElement('a');l.href=URL.createObjectURL(new Blob([JSON.stringify({rater:S.r,answers:S.a})]));l.download='answers_'+S.r+'.json';l.click()}
if(S.r){r.value=S.r}</script></body></html>"""

make() if a.mode == "make" else score()
