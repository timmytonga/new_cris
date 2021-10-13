my_run_expt.py --batch_size=64 --confounder_names=forest2water2 --dataset=CUB --gpu=1 --log_dir=AUTO \
--log_every=50 --lr=1e-3 --metadata_path=results/CUB/waterbird_sweep/metadata_aug.csv \
--n_epochs=100 --project_name=spurious --root_dir=./cub \
--save_step=10 --shift_type=confounder --target_name=waterbird_complete95 \
--val_fraction=0.2 --weight_decay=1e-4 --part 1 --part1_split_proportion 0.5\
--seed 1