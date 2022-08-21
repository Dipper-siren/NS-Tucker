#!/bin/bash
python train.py --task_dir=KG_Data/FB15K-237 --sample=IS --model=Tucker --loss=point --save=True --s_epoch=500 --hidden_dim=200 --lamb=0.01 --lr=0.0005 --n_epoch=3000 --n_batch=4096 --filter=True --epoch_per_test=10 --test_batch_size=60 --optim=adam --out_file=_base;

