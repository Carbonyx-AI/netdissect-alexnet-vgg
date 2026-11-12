#!/bin/bash
# Second round of experiments (AlexNet + VGG, paper models via Caffe). A job is started only when
# enough GPU memory and RAM are free, and at most MAXJ jobs run at the same time.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}; OUT=${ND_OUT:-results}; T=${ND_TRAINED:-work/trained}
MAXJ=${MAXJ:-2}; GPU_FREE_MB=${GPU_FREE_MB:-14000}; RAM_FREE_GB=${RAM_FREE_GB:-80}
LOG=${LOG:-logs}; mkdir -p $LOG

ok_to_start() {
  local g=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  local r=$(free -g | awk '/Mem:/{print $7}')
  [ "$(jobs -rp | wc -l)" -lt "$MAXJ" ] && [ "$g" -ge "$GPU_FREE_MB" ] && [ "$r" -ge "$RAM_FREE_GB" ]
}
launch() {   # launch <name> <command...>  (skipped if results/<name>/summary.json or logs/<name>.done exists)
  local n=$1; shift
  [ -f $OUT/$n/summary.json ] || [ -f $LOG/$n.done ] && return
  until ok_to_start; do sleep 20; done
  echo ">>> $(date +%T) $n"
  ( "$@" > $LOG/$n.log 2>&1 && touch $LOG/$n.done || echo "FAILED $n" ) &
  sleep 5
}
D() { local n=$1; shift; launch $n $PY run_dissect.py --name $n --gpu-mem-gb 10 "$@"; }
AL=conv1,conv2,conv3,conv4,conv5; VG=conv1_2,conv2_2,conv3_3,conv4_3,conv5_3

# ---- 0. reproducibility check against the first session
D check-alexnet-imagenet --model alexnet-imagenet --layers conv5
wait
# ---- 1. all layers of every model (paper Fig 5, 7)
for m in caffe-alexnet-places205 caffe-alexnet-imagenet caffe-alexnet-hybrid caffe-alexnet-places365; do D $m --model $m --layers $AL; done
for m in caffe-vgg16-places365 caffe-vgg16-hybrid1365 vgg16-imagenet vgg16bn-imagenet vgg16-random; do D $m --model $m --layers $VG; done
# ---- 2. rotation (paper Fig 4): the paper's model, AlexNet-ImageNet seeds 1-2, VGG16-Places365
R=0,0.2,0.4,0.6,0.8,1
for s in 0 1 2; do D rot-caffe-alexnet-places205-s$s --model caffe-alexnet-places205 --layers conv5 --rotate $R --rot-seed $s; done
for s in 1 2; do D rot-alexnet-imagenet-s$s --model alexnet-imagenet --layers conv5 --rotate $R --rot-seed $s; done
for s in 0 1 2; do D rot-caffe-vgg16-places365-s$s --model caffe-vgg16-places365 --layers conv5_3 --rotate $R --rot-seed $s; done
# ---- 3. method evaluation on the paper's model and on VGG
for m in caffe-alexnet-places205:conv5 caffe-vgg16-places365:conv5_3 vgg16-imagenet:conv5_3; do
  mm=${m%%:*}; l=${m##*:}
  D neg-$mm --model $mm --layers $l --negative
  for q in 0.01 0.001; do D q$q-$mm --model $mm --layers $l --quantile $q; done
  for z in 0.05 0.1; do D noise$z-$mm --model $mm --layers $l --noise $z; done
done
for f in fc conv5 conv4 conv3 conv2 conv1; do D cascade-$f-caffe-alexnet-places205 --model caffe-alexnet-places205 --layers conv5 --randomize-from $f; done
for f in fc conv5_1 conv4_1 conv3_1 conv2_1 conv1_1; do D cascade-$f-caffe-vgg16-places365 --model caffe-vgg16-places365 --layers conv5_3 --randomize-from $f; done
# ---- 4. training conditions (Sec 3.5) and width (Sec 3.7), 3 seeds each, Imagenette
for v in baseline nodropout batchnorm gap256 gap768 gap1024 gap2048; do for s in 0 1 2; do
  [ -f $T/$v-s$s/train_log.json ] || launch train-$v-s$s $PY train_small.py --variant $v --seed $s
done; done
wait
for v in baseline nodropout batchnorm gap256 gap768 gap1024 gap2048; do for s in 0 1 2; do
  ck=$(ls -v $T/$v-s$s/iter*.pt | tail -1); D small-$v-s$s --model "small-$v@$ck" --layers conv5
done; done
for s in 0 1 2; do for ck in $(ls -v $T/baseline-s$s/iter*.pt); do
  D small-baseline-s$s-$(basename $ck .pt) --model "small-baseline@$ck" --layers conv5
done; done
wait
echo "PHASE2 DONE $(date)"
