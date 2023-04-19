#!/bin/sh

python infer.py \
  -bs=512 \
  -lr=1e-4 \
  -ep=100 \
  -bi \
  -mod tv \
  --glove-emo-path="../data/glove.emotions.840B.300d.pt" \
  --hidden-sizes 300 200 100 \
  --num-layers=2 \
  --dataset=mosei_emo \
  --data-seq-len=20 \
  --aligned \
  --video-id='0' \
  --ckpt="tv.pt"