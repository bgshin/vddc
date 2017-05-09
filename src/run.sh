#!/bin/bash
export PYTHONPATH='../'
nohup python train_multi.py --w2vtrn 1 --train_dir train_1 > ./logs/multi_w2v_yelp_1.txt &
nohup python train_multi.py --w2vtrn 0 --train_dir train_0 > ./logs/multi_w2v_yelp_0.txt &