"""Pictures of units.

units   : 8 most activating Broden images per unit with the unit mask (= prototypes, Molnar), and
          the top-100 images where the assigned concept is least present (= criticisms).
          --human-study N also writes a blind rating page (paper Table 3) with true and decoy labels.
featviz : activation maximisation (Fourier image parameterisation + jitter/scale/rotation).
score   : score the answer files of the human study.
  python visualize.py units --run alexnet-places365 --model alexnet-places365 --layer conv5 --human-study 30
  python visualize.py featviz --run alexnet-imagenet --model alexnet-imagenet --layer conv5
  python visualize.py score --key results/.../human_study_key.json answers_*.json
"""
import argparse
import base64
import io
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import netdissect as ND
from netdissect import dissect as D
from netdissect.data import Broden
from netdissect.models import MEAN, STD, FeatureExtractor, arch_of, get_layer, load_model, preprocess

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["units", "featviz", "score"])
ap.add_argument("--run")
ap.add_argument("--model")
ap.add_argument("--layer")
ap.add_argument("--units", default=None, help="comma list, default: 12 best units")
ap.add_argument("--human-study", type=int, default=0)
ap.add_argument("--steps", type=int, default=512)
ap.add_argument("--key")
ap.add_argument("answers", nargs="*")
ap.add_argument("--data", default=ND.DATA)
ap.add_argument("--weights", default=ND.WEIGHTS)
ap.add_argument("--results", default=ND.OUT)
a = ap.parse_args()


def overlay(img, mask):
    """dim everything outside the mask, yellow border around it."""
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(img.shape[1::-1], Image.NEAREST)) > 0
    out = img.astype(np.float32) * 0.25
    out[m] = img[m]
    out[m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)] = [255, 220, 0]
    return out.astype(np.uint8)


def grid(images, ncol, pad=2):
    h, w = images[0].shape[:2]
    g = np.full((int(np.ceil(len(images) / ncol)) * (h + pad) - pad, ncol * (w + pad) - pad, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, ncol)
        g[r * (h + pad): r * (h + pad) + h, c * (w + pad): c * (w + pad) + w] = im
    return g


@torch.no_grad()
def units_mode():
    dev = ND.device(8.0)
    rdir = os.path.join(a.results, a.run)
    vdir = os.path.join(rdir, f"viz_{a.layer}")
    os.makedirs(vdir, exist_ok=True)
    fx = FeatureExtractor(load_model(a.model, a.weights), arch_of(a.model), [a.layer]).to(dev)
    T = torch.load(os.path.join(rdir, "thresholds.pt"))["T"][a.layer].to(dev)
    tally = pd.read_csv(os.path.join(rdir, f"tally_{a.layer}.csv")).set_index("unit")
    data = Broden(a.data, batch_size=64)
    meta = data.meta()
    num = {n: i for i, n in enumerate(meta["names"]) if n}
    maxact = torch.cat([fx(fx.normalize(imgs.to(dev)))[a.layer].amax((2, 3)).cpu() for imgs, _ in data.iter()]).numpy()
    order = np.argsort(-maxact, 0)                 # images sorted by max activation, per unit
    start, jump = fx.geometry(a.layer)

    def masks(u, idxs):
        imgs = torch.from_numpy(np.stack([data.images[i] for i in idxs])).to(dev)
        f = fx(fx.normalize(imgs))[a.layer][:, u:u + 1]
        g = D.rf_grid(f.shape[2], f.shape[3], start, jump, data.seg.shape[-1], imgs.shape[1], dev)
        return (D.upsample_rf(f, g)[:, 0] > T[u]).cpu().numpy()

    def concept_mask(i, c):
        if meta["cat_names"][meta["label_cat"][c]] in ("scene", "texture"):
            return np.full(data.seg.shape[-2:], bool(data.imglab[i, c]))
        return (np.asarray(data.seg[i]) == c).any(0)

    rows = []
    for u in [int(x) for x in a.units.split(",")] if a.units else tally.index[:12].tolist():
        top = order[:100, u]
        M = masks(u, top)
        lab = tally.loc[u, "label"]
        ov = np.array([(M[j] & concept_mask(i, num[lab])).sum() / max(1, M[j].sum()) for j, i in enumerate(top)])
        protos = [overlay(np.asarray(data.images[i]), M[j]) for j, i in enumerate(top[:8])]
        crits = [overlay(np.asarray(data.images[top[j]]), M[j]) for j in np.argsort(ov, kind="stable")[:4]]
        Image.fromarray(grid(protos, 8)).save(os.path.join(vdir, f"unit{u:04d}_prototypes.jpg"), quality=90)
        Image.fromarray(grid(crits, 4)).save(os.path.join(vdir, f"unit{u:04d}_criticisms.jpg"), quality=90)
        rows.append({"unit": u, "label": lab, "category": tally.loc[u, "category"], "iou": tally.loc[u, "iou"],
                     "frac_top100_mask_on_concept": float((ov > 0).mean()), "mean_overlap_top100": float(ov.mean())})
    pd.DataFrame(rows).to_csv(os.path.join(vdir, "prototype_criticism_stats.csv"), index=False)

    if a.human_study:              # half true labels, half decoys from the same category
        rng = np.random.default_rng(2017)
        det = tally[tally.is_detector == 1]
        items, key = [], []
        for n, u in enumerate(rng.choice(det.index.values, size=min(a.human_study, len(det)), replace=False)):
            top = order[:15, u]                    # 15 images per unit as in the paper
            M = masks(int(u), top)
            buf = io.BytesIO()
            Image.fromarray(grid([overlay(np.asarray(data.images[i]), M[j]) for j, i in enumerate(top)], 5)).save(buf, "JPEG", quality=85)
            true, cat = det.loc[u, "label"], det.loc[u, "category"]
            pool = sorted(set(tally[(tally.category == cat) & (tally.label != true)].label.dropna())) or \
                [meta["names"][c] for c in np.nonzero(meta["label_valid"] & (meta["label_cat"] == meta["cat_names"].index(cat)))[0]
                 if meta["names"][c] != true]
            use_true = bool(rng.random() < 0.5)
            shown = true if use_true else str(rng.choice(pool))
            items.append({"id": n, "img": base64.b64encode(buf.getvalue()).decode(), "phrase": shown.replace("-c", "").replace("-s", "")})
            key.append({"id": n, "unit": int(u), "true_label": true, "category": cat, "shown": shown,
                        "is_true_label": use_true, "iou": float(det.loc[u, "iou"])})
        json.dump(key, open(os.path.join(vdir, "human_study_key.json"), "w"), indent=1)
        open(os.path.join(vdir, "human_study.html"), "w").write(HTML.replace("__ITEMS__", json.dumps(items)))


def featviz_mode():
    dev = ND.device(8.0)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = load_model(a.model, a.weights).to(dev)
    model.requires_grad_(False)
    li = get_layer(model, arch_of(a.model), a.layer)
    tally = pd.read_csv(os.path.join(a.results, a.run, f"tally_{a.layer}.csv")).set_index("unit")
    units = [int(u) for u in a.units.split(",")] if a.units else tally.index[:12].tolist()
    mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    # image parameterised in Fourier space with 1/f scaling (smoother images)
    fy, fx_ = torch.fft.fftfreq(224, device=dev)[:, None], torch.fft.rfftfreq(224, device=dev)[None, :]
    scale = 1.0 / torch.clamp(torch.sqrt(fx_ ** 2 + fy ** 2), min=1.0 / 224)
    spec = (torch.randn(len(units), 3, 224, 113, 2, device=dev) * 0.01).requires_grad_(True)
    render = lambda: torch.sigmoid(torch.fft.irfft2(torch.view_as_complex(spec) * scale, s=(224, 224), norm="ortho") / 4.0)
    opt = torch.optim.Adam([spec], lr=0.05)
    uidx = torch.tensor(units, device=dev)
    for step in range(a.steps):
        img = render()
        # random jitter, scale (+-10%) and rotation (+-10 deg) at every step
        x = F.pad(img, (12, 12, 12, 12), mode="reflect")
        dy, dx = torch.randint(0, 25, (2,), generator=gen).tolist()
        x = x[:, :, dy:dy + 224, dx:dx + 224]
        s = 1.0 + (torch.rand(1, generator=gen).item() - 0.5) * 0.2
        th = (torch.rand(1, generator=gen).item() - 0.5) * 2 * np.deg2rad(10)
        A = torch.tensor([[np.cos(th) / s, -np.sin(th) / s, 0], [np.sin(th) / s, np.cos(th) / s, 0]],
                         dtype=x.dtype, device=dev)[None].expand(len(units), -1, -1)
        x = F.grid_sample(x, F.affine_grid(A, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
        dy, dx = torch.randint(-2, 3, (2,), generator=gen).tolist()
        h = torch.roll(x, (dy, dx), (2, 3))
        h = (h - mean) / std if not hasattr(model, "preprocess") else preprocess(model, h)
        for mod in model.features[: li.conv_idx + 1]:
            h = mod(h)
        act = h[torch.arange(len(units)), uidx].mean((1, 2))      # each image maximises its own unit
        tv = (img[:, :, 1:] - img[:, :, :-1]).abs().mean() + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
        loss = -act.sum() + 1e-3 * tv * len(units)
        opt.zero_grad(); loss.backward(); opt.step()
    out = os.path.join(a.results, a.run, "featviz")
    os.makedirs(out, exist_ok=True)
    with torch.no_grad():
        img = render().permute(0, 2, 3, 1).cpu().numpy()
    for j, u in enumerate(units):
        Image.fromarray((img[j] * 255).astype(np.uint8)).save(os.path.join(out, f"unit{u:04d}.png"))
    pd.DataFrame({"unit": units, "label": [tally.loc[u, "label"] for u in units], "iou": [tally.loc[u, "iou"] for u in units],
                  "final_activation": act.tolist()}).to_csv(os.path.join(out, "featviz_units.csv"), index=False)


def score_mode():
    key = {k["id"]: k for k in json.load(open(a.key))}
    raters = [json.load(open(p)) for p in a.answers]
    pooled = {True: [], False: []}
    for r in raters:
        res = {t: [r["answers"][str(i)] == "yes" for i, k in key.items() if k["is_true_label"] == t and str(i) in r["answers"]]
               for t in (True, False)}
        pooled[True] += res[True]; pooled[False] += res[False]
        print(f"{r['rater']}: true labels judged descriptive {np.mean(res[True]):.0%}, decoys {np.mean(res[False]):.0%}")
    print(f"pooled: true {np.mean(pooled[True]):.0%}, decoy {np.mean(pooled[False]):.0%}")
    ids = [i for i in key if all(str(i) in r["answers"] for r in raters)]
    if len(raters) > 1 and ids:            # Fleiss' kappa
        M = np.array([[sum(r["answers"][str(i)] == v for r in raters) for v in ("yes", "no")] for i in ids])
        n, p = len(raters), M.sum(0) / M.sum()
        Pbar, Pe = ((M * (M - 1)).sum(1) / (n * (n - 1))).mean(), (p ** 2).sum()
        print(f"Fleiss kappa ({len(ids)} items, {n} raters): {(Pbar - Pe) / (1 - Pe):.3f}")


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Unit Labelling Study</title>
<style>body{font:16px sans-serif;max-width:900px;margin:auto;padding:16px} img{width:100%}
.c{border:1px solid #ccc;border-radius:8px;padding:12px;margin:12px 0} .s{background:#2563eb;color:#fff}</style></head><body>
<h2>Unit Labelling Study</h2><p>Each panel: the 15 images that activate one CNN unit most; bright regions are where it fires.
Answer <b>yes</b> if the phrase describes most highlighted regions. Work alone. Then press Download.</p>
Name: <input id="r"><div id="items"></div><button onclick="dl()">Download answers</button>
<script>const I=__ITEMS__,A={};I.forEach(t=>{const d=document.createElement('div');d.className='c';
d.innerHTML=`<p>${t.id+1}. Does <b>"${t.phrase}"</b> describe most highlighted regions?</p><img src="data:image/jpeg;base64,${t.img}"><p><button>yes</button> <button>no</button></p>`;
d.querySelectorAll('button').forEach(b=>b.onclick=()=>{A[t.id]=b.textContent;d.querySelectorAll('button').forEach(x=>x.className=x===b?'s':'')});
document.getElementById('items').appendChild(d)});
function dl(){const r=document.getElementById('r').value||'anon',l=document.createElement('a');
l.href=URL.createObjectURL(new Blob([JSON.stringify({rater:r,answers:A})]));l.download='answers_'+r+'.json';l.click()}</script></body></html>"""

{"units": units_mode, "featviz": featviz_mode, "score": score_mode}[a.mode]()
