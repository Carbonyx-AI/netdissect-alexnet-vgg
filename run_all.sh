#!/bin/bash
# Everything, in order. Each dissection is skipped if its summary.json already exists.
#   bash run_all.sh            (needs ~25 GB disk in $ND_WORK; ~3 h on one GPU)
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
export ND_WORK=${ND_WORK:-$PWD/work}
OUT=${ND_OUT:-results}; T=$ND_WORK/trained; W=$ND_WORK
mkdir -p $W

# 1. data: Broden (official site is down -> Internet Archive copy, checked with SHA-1), Places365 AlexNet, Imagenette
if [ ! -d $W/broden ]; then
  cd $W
  curl -fSL --connect-timeout 20 -o broden1_224.zip http://netdissect.csail.mit.edu/data/broden1_224.zip || \
    curl -fSL -o broden1_224.zip "https://web.archive.org/web/20240707052113id_/https://netdissect.csail.mit.edu/data/broden1_224.zip"
  $PY -c "import hashlib,base64; h=base64.b32encode(hashlib.sha1(open('broden1_224.zip','rb').read()).digest()).decode(); assert h=='O5CPU2QYZGJJO5E3VXZSX6Z2IHFZSNPH', h"
  unzip -q broden1_224.zip
  curl -fSL -o alexnet_places365.pth.tar http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar
  curl -fSL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz | tar xz
  cd - > /dev/null
  $PY -m netdissect.data --root $W/broden1_224 --out $W/broden
fi

run() { local n=$1; shift; [ -f $OUT/$n/summary.json ] || nice $PY run_dissect.py --name $n "$@" || echo "FAILED $n"; }
AL=conv1,conv2,conv3,conv4,conv5
# 2. paper: layers (Fig 5), architectures / training data / random (Fig 7), BN (Sec 3.5), rotation (Fig 4)
run alexnet-imagenet  --model alexnet-imagenet  --layers $AL
run alexnet-places365 --model alexnet-places365 --layers $AL
run alexnet-random    --model alexnet-random    --layers $AL
run vgg16-imagenet    --model vgg16-imagenet    --layers conv3_3,conv4_3,conv5_3
run vgg16bn-imagenet  --model vgg16bn-imagenet  --layers conv3_3,conv4_3,conv5_3
for s in 0 1 2; do run rot-alexnet-places365-s$s --model alexnet-places365 --layers conv5 --rotate 0,0.2,0.4,0.6,0.8,1 --rot-seed $s; done
run rot-alexnet-imagenet-s0 --model alexnet-imagenet --layers conv5 --rotate 0,0.2,0.4,0.6,0.8,1
# 3. evaluation of the method: negative activations, thresholds, noise, cascading randomisation
run neg-alexnet-imagenet  --model alexnet-imagenet  --layers conv5 --negative
run neg-alexnet-places365 --model alexnet-places365 --layers conv5 --negative
run neg-vgg16-imagenet    --model vgg16-imagenet    --layers conv5_3 --negative
for m in alexnet-imagenet alexnet-places365; do
  for q in 0.01 0.001; do run q$q-$m --model $m --layers conv5 --quantile $q; done
  for n in 0.05 0.1; do run noise$n-$m --model $m --layers conv5 --noise $n; done
done
for f in fc conv5 conv4 conv3 conv2 conv1; do run cascade-$f-alexnet-imagenet --model alexnet-imagenet --layers conv5 --randomize-from $f; done
nice $PY run_fidelity.py --model alexnet-imagenet --layer conv5
nice $PY run_fidelity.py --model vgg16-imagenet --layer conv5_3
nice $PY visualize.py units --run alexnet-places365 --model alexnet-places365 --layer conv5 --human-study 30
nice $PY visualize.py units --run alexnet-imagenet --model alexnet-imagenet --layer conv5
nice $PY visualize.py units --run vgg16-imagenet --model vgg16-imagenet --layer conv5_3
nice $PY visualize.py featviz --run alexnet-imagenet --model alexnet-imagenet --layer conv5
nice $PY visualize.py featviz --run alexnet-places365 --model alexnet-places365 --layer conv5
nice $PY probes.py net2vec --run alexnet-imagenet --model alexnet-imagenet
nice $PY probes.py net2vec --run alexnet-places365 --model alexnet-places365
# 4. paper Sec 3.5 / 3.7 at small scale: train AlexNet variants on Imagenette, dissect them
V="baseline repeat1 repeat2 nodropout batchnorm gapwide"
for v in $V; do [ -f $T/$v/train_log.json ] || nice $PY train_small.py --variant $v; done
for v in $V; do run small-$v --model "small-$v@$(ls -v $T/$v/iter*.pt | tail -1)" --layers conv5; done
for ck in $(ls -v $T/baseline/iter*.pt); do run small-baseline-$(basename $ck .pt) --model "small-baseline@$ck" --layers conv5; done
mkdir -p $OUT/trained_logs; for v in $V; do cp $T/$v/train_log.json $OUT/trained_logs/$v.json; done
# 5. paper Sec 3.6: discrimination vs interpretability
M=alexnet-random=alexnet-random:conv5,alexnet-imagenet=alexnet-imagenet:conv5,alexnet-places365=alexnet-places365:conv5
M=$M,vgg16-imagenet=vgg16-imagenet:conv5_3,vgg16bn-imagenet=vgg16bn-imagenet:conv5_3
for v in $V; do M="$M,small-$v=small-$v@$(ls -v $T/$v/iter*.pt | tail -1):conv5"; done
nice $PY probes.py discrimination --models "$M"
# 6. tables and figures
cp $W/broden1_224/label.csv $OUT/broden_label.csv
$PY analyze.py --compact
