#!/bin/bash

DATASET=cub
p="0.8"
part1_model_epochs="10 50 100"
part1_n_epochs=201
seed=0
part2_methods="--part2_reweight"  # can add --part2_loss_type group_dro
part2_n_epochs=301
gpu=1

# python scripts/${DATASET}_sweep.py --seed $seed -p $p --part1_model_epochs $part1_model_epochs $part2_methods --part2_use_pgl --part2_n_epochs $part2_n_epochs --run_test --part1_pgl_model_epoch 0 --run_part1 --part1_n_epochs $part1_n_epochs

for i in 0 1 2 3 4 10; do python scripts/${DATASET}_sweep.py --seed $seed -p $p --part1_model_epochs $part1_model_epochs $part2_methods --part2_use_pgl --part2_n_epochs $part2_n_epochs --run_test --part1_pgl_model_epoch $i --gpu $gpu; done
