#!/bin/bash

# debug
# CUDA_VISIBLE_DEVICES=1 python train.py data=debug

# original bash
CUDA_VISIBLE_DEVICES=1 python train.py data=selected
