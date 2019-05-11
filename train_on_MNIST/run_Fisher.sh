#!/bin/bash
set -e # Exit the script if an error happens

maxeps=2
l_r=0.01
b_s=1024

python train_Fisher.py --epochs $maxeps --lr $l_r --batch-size=$b_s #--resume
