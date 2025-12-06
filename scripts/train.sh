#!/bin/bash

# GPU0 reproduce
# GPU1 modified version with 40 epochs

CUDA_VISIBLE_DEVICES=1 python train.py data=selected
