#!/bin/sh

python main.py                     \
	-bs=512                        \
	-lr=1e-4                       \
	-ep=100                        \
	--model=eea                    \
	-bi                            \
	--hidden-sizes 300 200 100     \
	--num-layers=2                 \
	--dropout=0.15                 \
	--data-seq-len=20              \
	--dataset=mosei_emo            \
	--aligned                      \
	--loss=bce                     \
	--clip=10.0                    \
	--early-stop=8                 \
	-mod=tav                       \
	--patience=5