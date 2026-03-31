#!/bin/bash

# debug
# CUDA_VISIBLE_DEVICES=1 python train.py data=debug

# train on dadagp
# python train.py data=train_split

# train on leduc
# python train.py data=leduc

# combined dadagp and leduc for training
python train.py data=combined
