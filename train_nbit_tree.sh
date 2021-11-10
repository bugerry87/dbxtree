#!/bin/sh
TRAIN='-X ./data/train_index.txt -e 500 --steps_per_epoch=1000 --save_best_only --learning_rate=0.001'
VAL='-Y ./data/val_index.txt --validation_steps=20 --validation_freq=1'
TEST='-T ./data/test_index.txt --test_steps=100 --test_freq=1 --floor=0.0005'
#PERM='--permute 47 31 15 46 30 14 45 29 13 44 28 12 43 27 11 42 26 10 41 25 9 40 24 8 39 23 7 38 22 6 37 21 5 36 20 4 35 19 3 34 18 2 33 17 1 32 16 0 --bits_per_dim 16 16 16 0'
#PERM='--permute 0 16 32 1 17 33 2 18 34 3 19 35 4 20 36 5 21 37 6 22 38 7 23 39 8 24 40 9 25 41 10 26 42 11 27 43 12 28 44 13 29 45 14 30 46 15 31 47 --bits_per_dim 16 16 16 0'
#PERM='--permute 39 37 30 22 2 17 16 1 3 19 20 18 32 0 4 21 33 5 34 6 7 24 23 8 35 9 36 25 26 10 11 38 28 27 12 29 40 13 41 42 14 15 31 43 44 46 45 47'
#PERM='--sort_bits absolute --bits_per_dim 16 16 16 0'
PERM='--sort_bits absolute --keypoints=0.03 --bits_per_dim 14 14 10 0' #--spherical
#PERM='--permute 0 14 28 1 15 29 2 16 30 3 17 31 4 18 32 5 19 33 6 20 34 7 21 35 8 22 36 9 23 37 10 24 38 11 25 12 26 13 27 --keypoints=0.03 --bits_per_dim 14 14 10 0' #--spherical
PAYLOAD='--payload'
TRANSFORM='--scale 100 100 10 0'
MODEL='-d 2 -k 128 -C 4 -D 2 --branches uids pos pivots meta --activation softmax --loss regularized_crossentropy'
LOG='-v 2 --log_dir=./gandm/logs'
CHECKPOINT='--checkpoint=./logs/20211031-172915/ckpts/nbittree_start.hdf5'

#cd ~/gandm
#git checkout .
#git fetch
#git checkout master
#git pull
#git show -s
#cd ~

#export TF_ENABLE_AUTO_MIXED_PRECISION=0
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/home/bugerry1987/anaconda3/lib
#python train_nbit_tree.py $TRAIN $MODEL $LOG $PERM $TEST $TRANSFORM $CHECKPOINT #$VAL #$PAYLOAD 
-X ./data/train_index.txt -e 100 --steps_per_epoch=100 --save_best_only --learning_rate=0.001 \
-T ./data/test_index.txt --test_steps=100 --test_freq=1 --floor=0.0005 \
--sort_bits absolute --keypoints=0.03 --bits_per_dim 14 14 10 0 \
-d 2 -k 128 -C 4 -D 2 --branches uids pos pivots meta \
--activation softmax --loss regularized_crossentropy \
-v 2 --log_dir=./logs --checkpoint=./logs/20211031-172915/ckpts/nbittree_0414-1.902.hdf5
--training_state=./logs/20211031-172915/ckpts/nbittree_0413-1.902.train.pkl