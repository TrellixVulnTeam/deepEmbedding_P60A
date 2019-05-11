@echo off
set maxeps=10
set l_r=0.001
set b_s=128
python train_classification.py --epochs %maxeps% --lr %l_r% --batch-size=%b_s% --resume
pause