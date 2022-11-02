#!/bin/sh
TRAIN='-X ./data/train_index.txt -e 100 --steps_per_epoch=100 --save_best_only --learning_rate=0.001'
#VAL='-Y ./data/val_index.txt --validation_steps=20 --validation_freq=1'
#TEST='-T ./data/test_index.txt --test_steps=20 --test_freq=1 --floor=0.0005'
#KEYPOINTS='--keypoints=0.03'
MODEL='-k 4 -C 2 -D 1 --branches uids pos pivots --activation softmax --radius 0.006'
LOG='-v 1 --log_dir=./gandm/logs'
#CHECKPOINT='--checkpoint ./gandm/logs/20211031-172915/ckpts/nbittree_0414-1.902.hdf5 --training_state ./gandm/logs/20211031-172915/ckpts/nbittree_0413-1.902.train.pkl'

#cd ~/gandm
#git checkout .
#git fetch
#git checkout master
#git pull
#git show -s
#cd ~

#export TF_ENABLE_AUTO_MIXED_PRECISION=0
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/home/bugerry1987/anaconda3/lib
python gandm/train_dbx_tree.py $MODEL $LOG $TRAIN $VAL $TEST $KEYPOINTS
