# Research log, session 2 (2026-11-12)

## Data
- Broden broden1_224 re-downloaded from the Internet Archive (2024-07-07 capture); SHA-1 matches the archived
  original (O5CPU2QYZGJJO5E3VXZSX6Z2IHFZSNPH). Preprocessing gives the same result as session 1 (max 5 label layers).
- Stanford40 (Action40) from the Internet Archive (2015-09-18 capture); SHA-1 of both zips matches the archive record.
  9,532 images, 40 actions, official split 4,000 train / 5,532 test.
- Caffe models downloaded from CSAIL / BVLC servers: AlexNet-Places205 (places205CNN_iter_300000 = the paper's
  baseline checkpoint, "trained to 300,000 iterations"), AlexNet-Hybrid (hybridCNN_iter_700000), AlexNet-Places365,
  VGG16-Places365, VGG16-Hybrid1365, BVLC AlexNet (ImageNet). Not available: VGG16-Places205 (404), self-supervised
  AlexNets (NetDissect server offline).

## Converter verification
- Weight shapes read from the .caffemodel files match the prototxt (incl. grouped conv: conv2 has 48 input channels per group).
- ImageNet check (Imagenette val, 1000-way top-1):
  - aspect-preserving resize + per-channel mean: BVLC AlexNet 68.0% vs torchvision AlexNet 73.4% -> gap investigated
  - 224 vs 227 crop: 68.0 vs 68.3% -> not the cause
  - warped 256x256 resize (Caffe's training protocol): BVLC AlexNet 76.7% (mean image 76.3%), torchvision 72.0%,
    Caffe VGG16-Hybrid1365 78.5% -> converter correct; the gap was the resize protocol. Broden images are square,
    so dissection is unaffected; per-channel mean is enough.
- Scene check (Broden images whose scene label is a class of the model):
  PyTorch AlexNet-Places365 43.1% / 78.4% (top-1/top-5), Caffe AlexNet-Places365 46.6 / 80.7,
  Caffe AlexNet-Places205 62.0 / 92.4 (205 classes), Caffe VGG16-Places365 51.5 / 84.7, VGG16-Hybrid1365 50.9 / 85.0.
- Caffe AlexNet-Hybrid: 15.9 / 54.5 -> investigated: class mapping verified (index order and the Places part equal the
  Places205 list); for 63% of the scene images the top class is an ImageNet object class; restricted to its 205 scene
  outputs it reaches 38.9% (Places205 model: 62.0%). Functional, but weaker than expected -> used, with a caveat.

## Code changes
- netdissect/caffe.py (own protobuf reader + prototxt parser), Caffe models in the model registry, model-specific preprocessing.
- Finer candidate levels in the exact quantile (needed for VGG conv1_2: 3.2e9 activations per unit); results unchanged (exact algorithm).
- train_small.py: --seed; width variants gap256/768/1024/2048 (gap256 = control: GAP without widening).
- probes.py: Action40 with a one-vs-rest linear SVM, C=0.001 (paper Sec. 3.6).
- human_check.py: 10-minute human evaluation (paper Table 3) with decoys and automatic scoring.

## Phase 2 results as they come in
- Reproducibility: check-alexnet-imagenet conv5 (session 2 code, fresh download) is bit-identical to session 1
  (thresholds, all IoU sums, tally).
- Caffe AlexNet-Places205 (paper's model), unique detectors conv1..conv5: 7, 20, 34, 37, 63 (objects at conv5: 33).
  Caffe AlexNet-ImageNet (BVLC): 7, 20, 35, 29, 43 (objects 15).
  -> Places > ImageNet now clear (63 vs 43, objects 33 vs 15); with torchvision/PyTorch models it was 55 vs 53.
  -> Places205 rises monotonically as in paper Fig. 5. Our absolute counts are higher than the paper's (its Fig. 4/5
     y-axes end at 40); possible reasons: 224 instead of 227 input, other pipeline details. Reported, not explained away.
- Caffe AlexNet-Places365 has 0-1 "tie" units (zero on >99.5% of positions) per layer, while the PyTorch
  AlexNet-Places365 release had 237 of 384 in conv3 -> the many dead units were a property of that PyTorch checkpoint,
  not of Places365 training; this explains part of the weak Places365 results of session 1.
- Caffe AlexNet conv5 unique detectors: Places365 68, Hybrid 65, Places205 63, ImageNet 43 -> same ordering as the
  paper's Fig. 8 for AlexNet (Places365 > Hybrid > Places205 > ImageNet).
- Caffe AlexNet-Places365 conv3 and conv4 both have 40 unique concepts with identical category split: checked, the
  layers are distinct (10 concepts only in conv3, 10 only in conv4, different thresholds) -> coincidence.
- VGG16 Caffe, unique detectors conv1_2..conv5_3: Places365 6, 7, 10, 44, 148 (conv5_3: 59 objects, 48 scenes);
  Hybrid1365 6, 10, 13, 44, 147. VGG16-ImageNet (session 1) conv5_3 89 -> Places > ImageNet clearly for VGG too.
- VGG16-ImageNet and VGG16-BN conv3_3/conv4_3/conv5_3 are identical to session 1 (9/46/89 and 2/26/96).
- Rotation of the paper's model (Caffe AlexNet-Places205 conv5), seed 0: 63, 60, 57, 45, 18, 11 for alpha 0..1
  -> 82.5% drop at alpha=1 (paper: 80%). alpha=0 equals the baseline (63) -> check passed.
- Rotation of the paper's model, 3 seeds: alpha=1 gives 11, 10, 9 unique detectors (baseline 63) -> mean drop 84%
  (paper 80%). Reproduced.
- VGG16 random: 7, 7, 4, 5, 4 unique detectors (conv1_2..conv5_3) -> stays low as expected.
- Human check page: 50 items (8 per AlexNet-Places205 layer, 10 VGG16-Places365 conv5_3), 24 true / 26 decoy labels;
  per-layer true-label counts 2-5 -> per-layer numbers will be noisy, pooled numbers are the main result.
- Rotation torchvision AlexNet-ImageNet, 3 seeds: 53 -> 23, 22, 21 (58% drop), smaller than the paper model (84%).
  Open question: model (torchvision vs Caffe) or training data (ImageNet vs Places)? -> add rotation of Caffe
  AlexNet-ImageNet (3 seeds) to separate the two.
- Rotation Caffe VGG16-Places365 conv5_3: 148 -> 29 (s0), 32 (s1) at alpha=1 (~79% drop); alpha=0.2 can be slightly
  above baseline (151, s1) -> small rotations are noise-level, the collapse happens between 0.6 and 0.8 as in AlexNet.
- First full analysis (local): last layer of 12 models: Places > ImageNet inside each family (AlexNet Caffe: 63-68 vs 43;
  VGG16: 147-148 vs 89); VGG > AlexNet; random 4-6. Surprise: torchvision AlexNet-ImageNet (53) > BVLC Caffe (43).
  Threshold sweep 0.02-0.08: 8 of 66 model pairs change order.
- Cascade (paper model): none 63, fc 63, +conv5 10, +conv4 8, +conv3 2, +conv2 6, +conv1 2 -> sanity check passed.
- Small models trained (val. acc, 3 seeds): baseline 82.0-82.9, nodropout 77.8-78.9, batchnorm 86.2-86.3,
  gap256/768/1024/2048 82.3-83.3. baseline-s0 82.9 vs 82.6 in session 1 (GPU training is not bit-deterministic).
- Small models, unique detectors (s0,s1,s2): baseline 21,18,18; nodropout 19,16,17; batchnorm 28,21,22.
- Width control: gap256 (GAP head, no widening) 11,11,9 unique with 99-113 detector units vs baseline ~19 unique,
  ~60 units -> the GAP head itself changes the picture; session 1's "more detector units" for the wide model was
  (at least partly) the GAP head, not the width. Width must be judged within the GAP family (256->768->1024->2048).
- small-gap2048-s0 dissection: CUDA OOM at our own 10 GB cap (2048 units x batch 32 masks). Re-run with batch 8
  (results do not depend on batch size: exact IoU sums and exact quantile).
- Width (GAP family), unique / detector units: gap256 11,11,9 / 99-113; gap768 14,11,12 / 296-366;
  gap1024 13,14,13 / 401-459.
- Phase 3 bugs found and fixed: run_fidelity did not pass the weights dir (Caffe model not found); discrimination saved
  only at the end and crashed on the not-yet-dissected gap2048 models (now skips them and saves after every model).
- Action40 (linear SVM, mean per-class acc.; chance 2.5%): random 6.6-7.8%, AlexNets 30-35%, VGG16s 46-60%,
  Imagenette-trained 12-18%. Within AlexNet the Places models have more object detectors but lower Action40 accuracy
  than the ImageNet models (paper reports a similar mismatch for ResNet).
- Net2Vec, paper model conv5: directions beat the best unit for 87% of 61 concepts (mean IoU 0.076 vs 0.056).
- Rotation Caffe AlexNet-ImageNet s0: 43 -> 13 at alpha=1 (70% drop).
- Rotation Caffe AlexNet-ImageNet, 3 seeds: 43 -> 13, 12, 16 (68% drop). So ImageNet models drop less (58-68%) than
  Places models (69-84%), for both model families.
- gap2048 (batch 8 re-run): s0 17, s1 13 unique.
- Fidelity, Caffe AlexNet-ImageNet: pre-specified dog test NOT significant: 13 dog units, 1000-way drop 11.8 points vs
  10,000 random 13-unit groups mean 4.6, sd 7.2, max 49.6 -> p=0.12, z=1.0 (torchvision AlexNet: 24.3 points, p=1e-4).
  -> group fidelity of labels is model-dependent; the conclusion of session 1 is weakened.
