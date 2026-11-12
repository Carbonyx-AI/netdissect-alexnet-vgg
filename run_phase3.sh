#!/bin/bash
# After run_phase2.sh: pictures, probes, discrimination (Action40 + Broden scenes), human-check page.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}; OUT=${ND_OUT:-results}; T=${ND_TRAINED:-work/trained}
step() { echo ">>> $(date +%T) $*"; "$@" || echo "FAILED: $*"; }
for l in conv1 conv2 conv3 conv4 conv5; do step $PY visualize.py units --run caffe-alexnet-places205 --model caffe-alexnet-places205 --layer $l; done
step $PY visualize.py units --run caffe-vgg16-places365 --model caffe-vgg16-places365 --layer conv5_3
step $PY visualize.py featviz --run caffe-alexnet-places205 --model caffe-alexnet-places205 --layer conv5
step $PY visualize.py featviz --run caffe-vgg16-places365 --model caffe-vgg16-places365 --layer conv5_3
for m in caffe-alexnet-places205 caffe-vgg16-places365 vgg16-imagenet; do
  l=conv5; [[ $m == *vgg* ]] && l=conv5_3
  step $PY probes.py net2vec --run $m --model $m --layer $l
done
step $PY run_fidelity.py --model caffe-alexnet-imagenet --layer conv5
M=""
for m in alexnet-random alexnet-imagenet alexnet-places365 caffe-alexnet-imagenet caffe-alexnet-places205 caffe-alexnet-hybrid caffe-alexnet-places365; do M="$M,$m=$m:conv5"; done
for m in vgg16-random vgg16-imagenet vgg16bn-imagenet caffe-vgg16-places365 caffe-vgg16-hybrid1365; do M="$M,$m=$m:conv5_3"; done
for v in baseline nodropout batchnorm gap256 gap768 gap1024 gap2048; do for s in 0 1 2; do
  M="$M,small-$v-s$s=small-$v@$(ls -v $T/$v-s$s/iter*.pt | tail -1):conv5"; done; done
M=${M#,}
step $PY probes.py discrimination --task action40 --models "$M"
step $PY probes.py discrimination --task scenes --models "$M"
step $PY human_check.py make
echo "PHASE3 DONE $(date)"
