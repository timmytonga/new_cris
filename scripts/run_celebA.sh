LR=1e-5
WD=1e-5
#python my_run_expt.py --batch_size=64 --confounder_names=Male --dataset=CelebA \
#--gpu=1 --log_dir=AUTO \
#--log_every=50 --lr=$LR --metadata_path=results/celebA/metadata_aug.csv \
#--n_epochs=51 --project_name=splitpgl \
#--save_step=10 --shift_type=confounder --target_name=Blond_Hair \
# --weight_decay=$WD --part 1 --part1_split_proportion 0.5\
# --seed 1 --wandb
#
python run_expt.py --batch_size=64 --confounder_names=Male --dataset=CelebA \
--gpu=1 --log_dir=AUTO --root_dir=/home/thiennguyen/research/datasets/ \
--log_every=50 --lr=$LR --metadata_path=results/celebA/metadata_aug.csv \
--n_epochs=51 --project_name=splitpgl \
--save_step=10 --shift_type=confounder --target_name=Blond_Hair \
 --weight_decay=$WD --wandb