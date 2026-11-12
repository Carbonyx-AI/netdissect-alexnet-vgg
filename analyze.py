"""All tables (results/table_*.csv, results/results_summary.json) and figures (results/figures/).
Everything is computed from the saved IoU sums. Bootstrap = resampling the 50 image shards (last layers only).
Works without the preprocessed Broden (uses results/broden_label.csv) and with the shard-summed copies
sumsT_*.npz when the per-shard sums are missing (bootstrap columns are then NaN).
  python analyze.py             python analyze.py --compact   (also write sumsT_*.npz copies)
"""
import argparse
import glob
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import netdissect as ND  # noqa: E402
from netdissect import dissect as D  # noqa: E402
from netdissect.data import CATS, parse_labels  # noqa: E402

COL = {"object": "#2a78d6", "part": "#eb6834", "scene": "#1baf7a", "material": "#eda100", "texture": "#e87ba4", "color": "#008300"}
GREY = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=ND.OUT)
ap.add_argument("--data", default=ND.DATA)
ap.add_argument("--boot", type=int, default=200)
ap.add_argument("--compact", action="store_true")
a = ap.parse_args()
R, FIG = a.results, os.path.join(a.results, "figures")
os.makedirs(FIG, exist_ok=True)
if a.compact:
    for p in glob.glob(os.path.join(R, "*", "sums_*.npz")):
        q = p.replace(os.sep + "sums_", os.sep + "sumsT_")
        if not os.path.exists(q):
            d = np.load(p)
            np.savez_compressed(q, **{k: d[k].sum(0, keepdims=True) for k in d.files})
if os.path.exists(os.path.join(a.data, "meta.json")):
    info, m = json.load(open(os.path.join(a.data, "meta.json"))), np.load(os.path.join(a.data, "meta.npz"), allow_pickle=True)
    names, lc, valid = info["names"], m["label_cat"], (m["img_count"] >= 10) & (m["label_cat"] >= 0)
else:   # every label in label.csv has >= 10 images in broden1_224 (checked while preprocessing)
    names, lc, _ = parse_labels(os.path.join(R, "broden_label.csv"))
    valid = lc >= 0
valid[0] = False
meta = {"n_labels": len(names), "n_cats": len(CATS), "label_cat": np.where(lc >= 0, lc, 0), "cat_names": CATS,
        "names": names, "label_valid": valid}
AL, VG = ["conv1", "conv2", "conv3", "conv4", "conv5"], ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]
MODELS = [  # run, short label, layers
    ("alexnet-random", "AlexNet random", AL), ("alexnet-imagenet", "AlexNet ImageNet (tv)", AL),
    ("caffe-alexnet-imagenet", "AlexNet ImageNet (Caffe)", AL), ("alexnet-places365", "AlexNet Places365 (PT)", AL),
    ("caffe-alexnet-places365", "AlexNet Places365 (Caffe)", AL), ("caffe-alexnet-places205", "AlexNet Places205 (Caffe, paper)", AL),
    ("caffe-alexnet-hybrid", "AlexNet Hybrid (Caffe)", AL), ("vgg16-random", "VGG16 random", VG),
    ("vgg16-imagenet", "VGG16 ImageNet", VG), ("vgg16bn-imagenet", "VGG16-BN ImageNet", VG),
    ("caffe-vgg16-places365", "VGG16 Places365 (Caffe)", VG), ("caffe-vgg16-hybrid1365", "VGG16 Hybrid1365 (Caffe)", VG)]
LAST = {r: ls[-1] for r, _, ls in MODELS}
out = {}


def summ(run, v, iou=0.04):
    for f in (f"sums_{v}.npz", f"sumsT_{v}.npz"):
        p = os.path.join(R, run, f)
        if os.path.exists(p):
            s = dict(np.load(p))
            return D.summarize(s, meta, iou), s
    return None, None


count = lambda s: D.unique_detectors(s, meta)
NAN = {"unique_ci95": [np.nan, np.nan], "unique_boot_mean": np.nan, "label_agreement_mean": np.nan, "label_agreement_ci95": [np.nan, np.nan]}


def bootstrap(sums, B):
    S = sums["inter"].shape[0]
    if S == 1 or B == 0:
        return NAN
    rng, full = np.random.default_rng(0), D.summarize(sums, meta)
    det, uniq, agree = full["is_detector"], [], []
    for _ in range(B):
        s = D.summarize(sums, meta, shard_idx=rng.integers(0, S, S))
        uniq.append(count(s)["unique_total"])
        agree.append(float(np.mean(s["best"][det] == full["best"][det])) if det.any() else np.nan)
    return {"unique_ci95": [float(np.percentile(uniq, 2.5)), float(np.percentile(uniq, 97.5))],
            "unique_boot_mean": float(np.mean(uniq)), "label_agreement_mean": float(np.nanmean(agree)),
            "label_agreement_ci95": [float(np.nanpercentile(agree, 2.5)), float(np.nanpercentile(agree, 97.5))]}


def stacked(ax, labels, rows, title, err=None):
    x, bottom = np.arange(len(labels)), np.zeros(len(labels))
    for c in CATS:
        v = np.array([r[c] for r in rows], float)
        ax.bar(x, v, 0.62, bottom=bottom, color=COL[c], label=c, edgecolor="white", linewidth=0.8)
        bottom += v
    for xi, t in zip(x, bottom):
        ax.text(xi, t + 0.5, f"{t:.0f}" if abs(t - round(t)) < 1e-6 else f"{t:.1f}", ha="center", va="bottom", fontsize=7)
    if err is not None:
        ax.errorbar(x + 0.36, bottom, yerr=err, fmt="none", ecolor=GREY, elinewidth=1, capsize=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(bottom.max() * 1.15, 1) + 1)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_ylabel("Unique detectors")


def save(fig, name, ax=None, below=-0.16):
    if ax is not None:
        ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, below), handlelength=1.2, columnspacing=1.0)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


# ---------------- layers of every model (paper Fig. 5, 7) ----------------
tab = []
for run, lab, layers in MODELS:
    for l in layers:
        s, sums = summ(run, l)
        if s is None:
            continue
        u, K = count(s), len(s["best"])
        row = {"run": run, "model": lab, "layer": l, "units": K, **u["unique_per_cat"], "unique_total": u["unique_total"],
               "detector_units": u["detector_units"], "pct_detector_units": 100 * u["detector_units"] / K,
               "unmatched_units": int(K - u["detector_units"]), "mean_best_iou": float(s["best_iou"].mean()),
               **{f"boot_{k}": v for k, v in bootstrap(sums, a.boot if l == LAST[run] else 0).items()}}
        vc = pd.Series([names[c] for c in s["best"][s["is_detector"]]]).value_counts()
        row["most_common_concept"] = f"{vc.index[0]} ({vc.iloc[0]})" if len(vc) else ""
        row["n_dog_units"] = int(vc.get("dog", 0))
        tab.append(row)
L = pd.DataFrame(tab) if tab else pd.DataFrame(columns=["run", "layer", "unique_total"])
L.to_csv(os.path.join(R, "table_layers.csv"), index=False)
out["layers"] = L.to_dict("records")


def layer_fig(runs, fname, title):
    runs = [r for r in runs if r in set(L.run)]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(2.9 * len(runs), 2.8), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        d = L[L.run == r]
        stacked(ax, d.layer.tolist(), d.to_dict("records"), dict((m[0], m[1]) for m in MODELS)[r])
    np.atleast_1d(axes)[0].set_ylim(0, L[L.run.isin(runs)].unique_total.max() * 1.15 + 1)
    np.atleast_1d(axes)[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    save(fig, fname)


layer_fig(["caffe-alexnet-places205", "caffe-alexnet-imagenet", "alexnet-random"], "fig_layers_alexnet.png",
          "AlexNet layers (paper Fig. 5: Places205 and ImageNet, Caffe models)")
layer_fig(["caffe-vgg16-places365", "vgg16-imagenet", "vgg16bn-imagenet", "vgg16-random"], "fig_layers_vgg.png", "VGG16 layers")
lastrows = [L[(L.run == r) & (L.layer == LAST[r])].iloc[0].to_dict() for r, _, _ in MODELS if len(L[(L.run == r) & (L.layer == LAST[r])])]
if lastrows:
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    err = np.array([[r["unique_total"] - r["boot_unique_ci95"][0], r["boot_unique_ci95"][1] - r["unique_total"]] for r in lastrows]).T
    stacked(ax, [r["model"].replace(" (", "\n(").replace("AlexNet ", "AlexNet\n").replace("VGG16 ", "VGG16\n").replace("VGG16-BN ", "VGG16-BN\n")
                 for r in lastrows], lastrows, "Last conv layer of every model (paper Fig. 7), bootstrap 95% intervals",
            err=None if np.isnan(err).all() else err)
    ax.tick_params(axis="x", labelsize=5.8)
    save(fig, "fig_last_layer.png", ax, -0.3)

# ---------------- rotation (paper Fig. 4) ----------------
rot = {}
for run in sorted(glob.glob(os.path.join(R, "rot-*"))):
    sj = os.path.join(run, "summary.json")
    if not os.path.exists(sj):
        continue
    for v in json.load(open(sj))["variants"]:
        s, _ = summ(os.path.basename(run), v)
        rot.setdefault(os.path.basename(run).rsplit("-s", 1)[0], {}).setdefault(float(v.split("@rot")[1]), []).append(count(s))
rot_tab = [{"model": b, "alpha": al, "n_seeds": len(us), "unique_mean": float(np.mean([u["unique_total"] for u in us])),
            "unique_sd": float(np.std([u["unique_total"] for u in us])), "unique_min": min(u["unique_total"] for u in us),
            "unique_max": max(u["unique_total"] for u in us),
            "detector_units_mean": float(np.mean([u["detector_units"] for u in us])),
            **{c: float(np.mean([u["unique_per_cat"][c] for u in us])) for c in CATS}}
           for b, d in rot.items() for al, us in sorted(d.items())]
rot_df = pd.DataFrame(rot_tab)
rot_df.to_csv(os.path.join(R, "table_rotation.csv"), index=False)
out["rotation"] = rot_tab
out["rotation_drop_pct"] = {}
for b in rot:
    d = rot_df[rot_df.model == b]
    if {0.0, 1.0} <= set(d.alpha):
        u0, u1 = d[d.alpha == 0].unique_mean.iloc[0], d[d.alpha == 1].unique_mean.iloc[0]
        out["rotation_drop_pct"][b] = {"drop_pct": float(100 * (1 - u1 / u0)), "baseline": u0, "rotated_mean": u1,
                                       "rotated_sd": float(d[d.alpha == 1].unique_sd.iloc[0]), "n_seeds": int(d.n_seeds.max())}
if len(rot_df):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for b, c in zip(rot, ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]):
        d = rot_df[rot_df.model == b]
        ax.errorbar(d.alpha, d.unique_mean / d.unique_mean.iloc[0] * 100, yerr=d.unique_sd / d.unique_mean.iloc[0] * 100,
                    fmt="-o", color=c, ms=3, lw=1.8, capsize=2, label=f"{b.replace('rot-', '')} ({int(d.n_seeds.max())} seeds)")
    ax.axhline(20, color=GREY, ls=":", lw=1)
    ax.text(0.01, 22, "paper: 80% drop", fontsize=7, color=GREY)
    ax.set_xlabel("rotation α (0 = learned basis, 1 = random basis)"); ax.set_ylabel("unique detectors, % of α=0")
    ax.set_title("Random rotations of the last conv layer (paper Fig. 4)", fontsize=9, loc="left")
    ax.legend(fontsize=6.5)
    save(fig, "fig_rotation.png")
    b0 = "rot-caffe-alexnet-places205"
    if b0 in rot:
        d = rot_df[rot_df.model == b0]
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        stacked(ax, [f"{x:g}" for x in d.alpha], d.to_dict("records"), "AlexNet-Places205 conv5 rotated (3 seeds), paper Fig. 4")
        ax.set_xlabel("rotation α")
        save(fig, "fig_rotation_places205.png", ax, -0.3)

# ---------------- negative activations ----------------
neg = []
for run in ["alexnet-imagenet", "alexnet-places365", "caffe-alexnet-places205", "vgg16-imagenet", "caffe-vgg16-places365"]:
    l = LAST[run]
    sp, _ = summ(run, l)
    sn, _ = summ(f"neg-{run}", l)
    if sp is None or sn is None:
        continue
    up, un = count(sp), count(sn)
    both = sp["is_detector"] & sn["is_detector"]
    neg.append({"model": run, "layer": l, "pos_unique": up["unique_total"], "neg_unique": un["unique_total"],
                "pos_detector_units": up["detector_units"], "neg_detector_units": un["detector_units"],
                "both_same_concept": int(np.sum(sp["best"][both] == sn["best"][both])),
                **{f"pos_{c}": up["unique_per_cat"][c] for c in CATS}, **{f"neg_{c}": un["unique_per_cat"][c] for c in CATS},
                "top_negative_labels": pd.Series([names[c] for c in sn["best"][sn["is_detector"]]]).value_counts().head(5).to_dict()})
out["negative"] = neg

# ---------------- stability: IoU threshold, activation quantile, input noise, bootstrap ----------------
stab = []
for run in ["alexnet-imagenet", "alexnet-places365", "caffe-alexnet-places205", "vgg16-imagenet", "caffe-vgg16-places365"]:
    l = LAST[run]
    s0, sums0 = summ(run, l)
    if s0 is None:
        continue
    det0 = s0["is_detector"]
    for t in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        st = D.summarize(sums0, meta, t)
        stab.append({"model": run, "perturbation": "iou_threshold", "value": t, "unique": count(st)["unique_total"],
                     "detector_units": int(st["is_detector"].sum()), "label_agreement": float(np.mean(st["best"] == s0["best"]))})
    for kind, prefix, v in [("quantile", "q", 0.01), ("quantile", "q", 0.001), ("noise", "noise", 0.05), ("noise", "noise", 0.1)]:
        s1, _ = summ(f"{prefix}{v}-{run}", l)
        if s1 is not None:
            stab.append({"model": run, "perturbation": kind, "value": v, "unique": count(s1)["unique_total"],
                         "detector_units": count(s1)["detector_units"],
                         "label_agreement": float(np.mean(s1["best"][det0] == s0["best"][det0]))})
S = pd.DataFrame(stab)
S.to_csv(os.path.join(R, "table_stability.csv"), index=False)
out["stability"] = stab

# ---- does the ranking of models depend on the IoU threshold? (paper: "relative orderings remain stable")
thr = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
U = {}
for r, lab, _ in MODELS:
    _, sums = summ(r, LAST[r])
    if sums is not None:
        U[lab] = [count(D.summarize(sums, meta, t))["unique_total"] for t in thr]
if len(U) > 2:
    U = pd.DataFrame(U, index=thr).T
    U.to_csv(os.path.join(R, "table_threshold_sweep.csv"))
    pairs = [(x, y) for x, y in combinations(U.index, 2) if U.loc[x, 0.04] != U.loc[y, 0.04]]
    flips = [(x, y) for x, y in pairs if any(np.sign(U.loc[x, t] - U.loc[y, t]) == -np.sign(U.loc[x, 0.04] - U.loc[y, 0.04]) for t in thr)]
    out["threshold_ordering"] = {"n_pairs": len(pairs), "n_pairs_flipping": len(flips), "flipping_pairs": flips,
                                 "kendall_tau_0.02_vs_0.08": float(stats.kendalltau(U[0.02], U[0.08]).statistic)}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for lab in U.index:
        ax.plot(thr, U.loc[lab], "-o", ms=3, lw=1.3, label=lab, color="#2a78d6" if "AlexNet" in lab else "#eb6834",
                alpha=0.35 if "random" in lab else 0.9)
    ax.axvline(0.04, color=GREY, ls="--", lw=0.8)
    ax.set_xlabel("IoU threshold for a detector"); ax.set_ylabel("Unique detectors (last conv)"); ax.set_yscale("log")
    ax.set_title(f"Threshold sweep: {len(flips)} of {len(pairs)} model pairs change order", fontsize=9, loc="left")
    save(fig, "fig_threshold_sweep.png")

# ---------------- cascading randomisation ----------------
casc = {}
for run, order in [("alexnet-imagenet", ["fc", "conv5", "conv4", "conv3", "conv2", "conv1"]),
                   ("caffe-alexnet-places205", ["fc", "conv5", "conv4", "conv3", "conv2", "conv1"]),
                   ("caffe-vgg16-places365", ["fc", "conv5_1", "conv4_1", "conv3_1", "conv2_1", "conv1_1"])]:
    s, _ = summ(run, LAST[run])
    if s is None:
        continue
    rows = [{"randomized": "none", **count(s)}]
    for f in order:
        sc, _ = summ(f"cascade-{f}-{run}", LAST[run])
        if sc is not None:
            rows.append({"randomized": f, **count(sc), "label_agreement_with_trained": float(np.mean(sc["best"] == s["best"]))})
    casc[run] = rows
out["cascade"] = casc
if casc:
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for (run, rows), c in zip(casc.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.plot(range(len(rows)), [r["unique_total"] for r in rows], "-o", color=c, ms=4, lw=1.8, label=run)
    ax.set_xticks(range(7), ["none", "fc only", "+top conv", "+4th", "+3rd", "+2nd", "+1st (all)"], fontsize=7)
    ax.set_ylabel("Unique detectors (last conv)"); ax.set_xlabel("layers re-initialised, from the top")
    ax.set_title("Cascading randomisation sanity check", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    save(fig, "fig_cascade.png")

# ---------------- consistency of concept sets (Jaccard, last layers) ----------------
sets = {lab: set(s["best"][s["is_detector"]].tolist()) for r, lab, _ in MODELS if (s := summ(r, LAST[r])[0]) is not None}
J = pd.DataFrame([[len(sets[x] & sets[y]) / max(1, len(sets[x] | sets[y])) for y in sets] for x in sets], index=list(sets), columns=list(sets))
J.to_csv(os.path.join(R, "table_jaccard.csv"))
out["jaccard"] = J.round(3).to_dict()

# ---------------- fidelity, Net2Vec, verification, human check ----------------
out["fidelity"] = {os.path.relpath(p, R): json.load(open(p)) for p in glob.glob(os.path.join(R, "*", "fidelity_*.json"))}
out["net2vec"] = {os.path.relpath(p, R): json.load(open(p)) for p in glob.glob(os.path.join(R, "*", "net2vec_*.json"))}
for f in ["verify_imagenette.json", "verify_scenes.json", "human_check.json"]:
    if os.path.exists(os.path.join(R, f)):
        out[f[:-5]] = json.load(open(os.path.join(R, f)))
for p in glob.glob(os.path.join(R, "*", "fidelity_units_*.csv")):
    df, tag = pd.read_csv(p), os.path.basename(os.path.dirname(p))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for det, c, lab in [(1, "#2a78d6", "detector (IoU>0.04)"), (0, "#eb6834", "no concept")]:
        ax.scatter(df[df.is_detector == det].iou, df[df.is_detector == det].imp_loss_ratio, s=10, color=c, alpha=0.8, label=lab)
    ax.set_xlabel("IoU of unit's best concept"); ax.set_ylabel("Permutation importance (loss ratio)")
    ax.set_title(f"{tag}: interpretability vs importance", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    save(fig, f"fig_fidelity_{tag}.png")

# ---------------- training conditions and width, 3 seeds (paper Sec. 3.5, 3.7) ----------------
VARS = ["baseline", "nodropout", "batchnorm", "gap256", "gap768", "gap1024", "gap2048"]
small = []
for v in VARS:
    for k in range(3):
        s, _ = summ(f"small-{v}-s{k}", "conv5")
        if s is None:
            continue
        tl = next((p for p in (os.path.join(ND.TRAINED, f"{v}-s{k}", "train_log.json"), os.path.join(R, "trained_logs", f"{v}-s{k}.json"))
                   if os.path.exists(p)), None)
        small.append({"variant": v, "seed": k, "units": len(s["best"]), "val_acc": json.load(open(tl))["final_val_acc"] if tl else np.nan,
                      **count(s)["unique_per_cat"], "unique_total": count(s)["unique_total"], "detector_units": count(s)["detector_units"]})
SM = pd.DataFrame(small)
if len(SM):
    SM.to_csv(os.path.join(R, "table_training_seeds.csv"), index=False)
    agg = SM.groupby("variant", sort=False).agg(n=("seed", "size"), val_acc=("val_acc", "mean"), val_acc_sd=("val_acc", "std"),
                                                unique=("unique_total", "mean"), unique_sd=("unique_total", "std"),
                                                unique_min=("unique_total", "min"), unique_max=("unique_total", "max"),
                                                detector_units=("detector_units", "mean"), detector_units_sd=("detector_units", "std"),
                                                object=("object", "mean"), texture=("texture", "mean")).reset_index()
    base = SM[SM.variant == "baseline"].unique_total.values
    agg["welch_p_vs_baseline"] = [float(stats.ttest_ind(SM[SM.variant == v].unique_total, base, equal_var=False).pvalue)
                                  if v != "baseline" and len(base) > 1 else np.nan for v in agg.variant]
    agg.to_csv(os.path.join(R, "table_training_variants.csv"), index=False)
    out["training_variants"] = agg.to_dict("records")
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = np.arange(len(agg))
    ax.bar(x, agg.unique, 0.6, color="#2a78d6", yerr=agg.unique_sd, capsize=3, label="unique detectors")
    for i, v in enumerate(agg.variant):
        ax.scatter([i] * (SM.variant == v).sum(), SM[SM.variant == v].unique_total, color="black", s=8, zorder=3)
    ax.set_xticks(x, [f"{v}\n{100 * acc:.1f}%" for v, acc in zip(agg.variant, agg.val_acc)], fontsize=7)
    ax.set_ylabel("Unique detectors (conv5)")
    ax.set_title("AlexNet variants trained on Imagenette, 3 seeds each (x: variant, val. acc.)", fontsize=9, loc="left")
    save(fig, "fig_training_variants.png")
dyn = []
for r in glob.glob(os.path.join(R, "small-baseline-s*-iter*")):
    k, it = os.path.basename(r).split("-s")[1].split("-iter")
    s, _ = summ(os.path.basename(r), "conv5")
    if s is not None:
        dyn.append({"seed": int(k), "iteration": int(it), **count(s)["unique_per_cat"], "unique_total": count(s)["unique_total"]})
DY = pd.DataFrame(dyn)
if len(DY):
    DY.to_csv(os.path.join(R, "table_training_dynamics.csv"), index=False)
    g = DY.groupby("iteration").agg(["mean", "std"])
    out["training_dynamics"] = {int(i): {c: float(g.loc[i, (c, "mean")]) for c in ["unique_total", "object", "texture", "part", "color", "scene"]}
                                for i in g.index}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for c in ["object", "part", "texture", "color", "scene"]:
        ax.errorbar(np.maximum(g.index, 1), g[(c, "mean")], yerr=g[(c, "std")], fmt="-o", color=COL[c], ms=3, lw=1.6, capsize=2, label=c)
    ax.set_xscale("log"); ax.set_xlabel("training iteration (iteration 0 plotted at 1)"); ax.set_ylabel("Unique detectors")
    ax.set_title("Emergence during training (3 seeds, paper Fig. 10)", fontsize=9, loc="left")
    save(fig, "fig_training_dynamics.png", ax)

# ---------------- discrimination vs interpretability (paper Sec. 3.6, Fig. 12) ----------------
for task, f, key in [("action40", "discrimination_action40.json", "action40_acc"), ("scenes", "discrimination.json", "scene_probe_acc")]:
    p = os.path.join(R, f)
    if not os.path.exists(p):
        continue
    rows = json.load(open(p))["rows"]
    sp = lambda rs: {"rho": float(stats.spearmanr([r[key] for r in rs], [r["unique_object_detectors"] for r in rs]).statistic),
                     "p": float(stats.spearmanr([r[key] for r in rs], [r["unique_object_detectors"] for r in rs]).pvalue), "n": len(rs)}
    pre = [r for r in rows if not r["run"].startswith("small-")]
    out[f"discrimination_{task}"] = {"rows": rows, "all": sp(rows), "pretrained_and_random": sp(pre),
                                     "imagenette_trained": sp([r for r in rows if r["run"].startswith("small-")])}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for r in rows:
        sm = r["run"].startswith("small-")
        ax.scatter(r[key], r["unique_object_detectors"], s=14 if sm else 26, color="#9a9a9a" if sm else ("#2a78d6" if "alexnet" in r["run"] else "#eb6834"))
        if not sm:
            ax.annotate(r["run"].replace("caffe-", ""), (r[key], r["unique_object_detectors"]), fontsize=5.5, color=GREY,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"{task}: mean per-class accuracy of a linear classifier on last-conv features")
    ax.set_ylabel("Unique object detectors")
    ax.set_title(f"{task}: Spearman ρ = {out[f'discrimination_{task}']['pretrained_and_random']['rho']:.2f} "
                 f"(pretrained/random models, n={len(pre)}); grey = Imagenette-trained", fontsize=7.5, loc="left")
    save(fig, f"fig_discrimination_{task}.png")

json.dump(out, open(os.path.join(R, "results_summary.json"), "w"), indent=1,
          default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if isinstance(o, float) and np.isnan(o) else str(o)))
print(L[["run", "layer", "unique_total", "detector_units"] + CATS].to_string() if len(L) else "no runs found")
