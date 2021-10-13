if [ -z "$1" ]
then 
    LR=1e-3
else
    LR=$1
fi

N_EPOCHS=20
WD=1e-5
python my_run_expt.py --batch_size=64 --confounder_names=forest2water2 --dataset=CUB --gpu=0 \
--log_every=50 --lr=${LR} --metadata_path=results/CUB/waterbird_sweep/metadata_aug.csv \
--n_epochs=${N_EPOCHS} --project_name=splitpgl --root_dir=./cub \
--save_step=10 --shift_type=confounder --target_name=waterbird_complete95 \
--val_fraction=0.2 --weight_decay=${WD} --part 2 \
 --seed 1\
 --log_dir=/home/thiennguyen/research/pseudogroups/CUB/autolog_splitpgl/erm_upweight-1_epochs101_lr0.001_wd0.0001/part2 \
 --part1_model_epoch=100 --upweight=0 \
 --part2_only_last_layer --part2_use_old_model \
 --wandb  # upweight=0 means equal sampling of the groups


#    parser.add_argument("--part1_model_epoch", type=int, help="Specify which epoch to load the initial model1 from")
#    parser.add_argument("--part2_only_last_layer", action="store_true", default=False, help="This is used to finetune model1 last layer")
#    parser.add_argument("--part2_use_old_model", action="store_true", default=False, help="For part2: whether we want to resuse the model from part1")
