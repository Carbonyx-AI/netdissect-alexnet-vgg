# Network Dissection of AlexNet and VGG16 in PyTorch

Interpretable Machine Learning course project. Re-implementation of Bau, Zhou, Khosla, Oliva & Torralba,
*Network Dissection: Quantifying Interpretability of Deep Visual Representations*, CVPR 2017
(doi:10.1109/CVPR.2017.354), with an evaluation of the method based on C. Molnar,
*Interpretable Machine Learning* (https://christophm.github.io/interpretable-ml-book).

The dissection itself (quantile thresholds, receptive-field upsampling, IoU, labelling), the rotations,
permutation importance, Net2Vec probes and feature visualisation are written in this repository; no
existing Network Dissection / Lucid / Captum code is used. torchvision gives only the network definitions
and ImageNet weights; SciPy gives the Schur decomposition and the statistical tests.

## Install

Python >= 3.10, then `pip install -r requirements.txt`. A CUDA GPU is used when available (3-5 GB of GPU
memory is enough); otherwise everything runs on CPU (slowly).

## Quick check (5 minutes, no GPU, no dataset)

```bash
python tests/test_core.py   # IoU vs brute force, exact quantile with ties, rotations, receptive fields, hooks
python analyze.py           # rebuilds every table and figure in results/ from the saved run outputs
```
Without the per-shard sums (`results/*/sums_*.npz`, ~580 MB, not in the small zip) the bootstrap columns are
NaN; all other numbers are reproduced exactly from the shard-summed copies `sumsT_*.npz`.

## Full run (~3 h on one GPU, ~25 GB disk)

```bash
bash run_all.sh
```
It downloads Broden `broden1_224` (the official server was down during the project, so it falls back to the
Internet Archive copy and checks SHA-1 `O5CPU2QYZGJJO5E3VXZSX6Z2IHFZSNPH`), the CSAIL AlexNet-Places365
weights and Imagenette, preprocesses Broden, and runs every experiment of the report in order.
Paths: `ND_WORK` (default `./work`) for data, `ND_OUT` (default `./results`) for results; `ND_DEVICE=cpu`
forces the CPU. Single runs, e.g.
```bash
python run_dissect.py --model alexnet-places365 --layers conv1,conv2,conv3,conv4,conv5
python run_dissect.py --model alexnet-places365 --layers conv5 --rotate 0,0.2,0.4,0.6,0.8,1
python run_dissect.py --model alexnet-imagenet --layers conv5 --max-images 2000     # quick CPU test
```

## Files

| file | what it does |
|---|---|
| `netdissect/data.py` | Broden preprocessing (CSV, RG-encoded label PNGs, merging of label maps) and the loader |
| `netdissect/models.py` | models, layer names, feature extraction (pre/post ReLU), receptive fields, small AlexNet variants |
| `netdissect/dissect.py` | exact quantiles, upsampling, IoU, labelling, unique detectors, random/fractional rotations |
| `run_dissect.py` | one dissection run (rotation, negative, quantile, noise, cascading randomisation options) |
| `run_fidelity.py` | permutation importance of units on Imagenette + accuracy check for the rotation |
| `probes.py` | Net2Vec concept directions vs units; discrimination vs interpretability (paper Sec. 3.6) |
| `visualize.py` | prototypes/criticisms, feature visualisation, blind human-study page and its scoring |
| `train_small.py` | AlexNet variants trained on Imagenette (paper Sec. 3.5 / 3.7, reduced scale) |
| `analyze.py` | bootstrap, all tables and figures |
| `run_all.sh` | the whole pipeline |
| `tests/test_core.py` | correctness checks |
| `report/` | report (LaTeX source and PDF) |

Each run folder in `results/` has `log.txt`, `thresholds.pt`, `sums_<layer>.npz`, `tally_<layer>.csv`
(unit, label, category, IoU, detector) and `summary.json`. `results/table_*.csv`, `results/results_summary.json`
and `results/figures/` are written by `analyze.py`.

## Human study

Each rater opens `results/alexnet-places365/viz_conv5/human_study.html` alone (do not open the key file),
answers, downloads the answers file; then
`python visualize.py score --key results/alexnet-places365/viz_conv5/human_study_key.json answers_*.json`.

## Making the submission zip

```bash
zip -qr ../netdissect-submission.zip . -x "work/*" "results/*/sums_*.npz" "*__pycache__*"   # ~70 MB
```
