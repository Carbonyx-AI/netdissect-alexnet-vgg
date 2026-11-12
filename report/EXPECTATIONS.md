# Pre-registered expectations (written 2026-11-12, before running phase 2)

Models: Caffe AlexNet (BVLC ImageNet, Places205 = the paper's model, Hybrid, Places365), torchvision
AlexNet (ImageNet, random), PyTorch AlexNet-Places365; torchvision VGG16 (ImageNet, BN, random),
Caffe VGG16 (Places365, Hybrid1365). Converted Caffe models passed a classification check first
(BVLC AlexNet 76.7% vs torchvision 72.0% on Imagenette with warped inputs).

| # | Paper claim | My expectation | Counts as failure if |
|---|---|---|---|
| 1 | Rotation of AlexNet-Places205 conv5 removes ~80% of unique detectors (Sec 3.2) | 70-85% drop with the paper's own model | < 60% drop, or alpha=0 differs from baseline |
| 2 | Same with other models | VGG16-Places365 conv5_3 also drops > 50% | drop < 30% |
| 3 | Low layers colour/texture, conv5 objects/parts (Sec 3.3) | holds for every AlexNet and every VGG | objects not maximal at last layer |
| 4 | Places > ImageNet (Sec 3.4) | Caffe AlexNet-Places205 > Caffe AlexNet-ImageNet; VGG16-Places365 > VGG16-ImageNet | reversed in both architectures |
| 5 | VGG > AlexNet (Sec 3.4) | holds for each training set | reversed |
| 6 | Hybrid between/near Places (Fig 7) | Hybrid >= ImageNet | - (no strong prior) |
| 7 | Random < trained | random AlexNet and VGG16 have < 15 unique detectors at the last layer | >= 25 |
| 8 | Seeds converge (Sec 3.5) | 3 seeds within +-3 unique detectors | spread > 6 |
| 9 | No dropout: more texture, fewer object detectors | small effect at our scale; unsure | - |
| 10 | Batch norm lowers interpretability | earlier single run said the opposite; with 3 seeds I expect no consistent decrease | - |
| 11 | Width 768 > 256, then flat for 1024/2048 (Sec 3.7) | detector *units* grow with width; unique concepts maybe not | - |
| 12 | Discrimination correlates with interpretability (Sec 3.6, Action40) | positive rank correlation across pretrained models | negative |
| 13 | Orderings stable across IoU thresholds (Sec 2.2) | some pairwise orderings will flip when gaps are small | - |
| 14 | Human evaluation (Table 3) | dissection labels accepted clearly more often than decoys; acceptance rises with layer | decoys accepted as often as true labels |

Verification rules applied to every run: rotation alpha=0 must reproduce the baseline exactly; random
models must stay low; numbers of tie units and candidate memory are logged; a result that contradicts
the paper is re-checked (inputs, preprocessing, threshold) before it is reported.
