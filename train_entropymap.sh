#!/bin/sh
TRAIN='-X ./data/train_index.txt -e 100 --steps_per_epoch=150 --stop_patience=-1'
VAL='-Y ./data/val_index.txt --validation_steps=50 --validation_freq=1'
MODEL='-k 64'
LOG='-v 2 --log_dir=./gandm/logs'

export TF_ENABLE_AUTO_MIXED_PRECISION=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bugerry1987/anaconda3/lib
python gandm/train_entropymap.py $TRAIN $VAL $MODEL $LOG