#!/bin/sh
TRAIN='-X ./data/train_index.txt -e 100 --steps_per_epoch=150 --stop_patience=10'
VAL='-Y ./data/val_index.txt --validation_steps=50 --validation_freq=9'
TEST='-T ./data/test_index.txt --test_steps=25 --test_freq=9'
MODEL='-d 2 -k 64 -t 0 -c 4'
LOG='-v 2 --log_dir=./gandm/logs'

python gandm/train_nbit_tree.py $TRAIN $VAL $TEST $MODEL $LOG