gpu=0
model=grnet
extra_tag=grnet_test_long
h5_file_path=nfs50/nfs50_no_polarity.hdf5

if [ -d "logs/${extra_tag}" ]; then
    rm -r logs/${extra_tag}
fi

mkdir logs/${extra_tag}

nohup python -u -W ignore train_grnet.py \
    --model ${model} \
    --npoint 2048 \
    --gpu ${gpu} \
    --batch_size 32 \
    --max_epoch 150 \
    --lr 1e-4 \
    --weight_decay 0 \
    --log_dir logs/${extra_tag} \
    --h5_file_path datas/${h5_file_path} \
    --workers 4 \
    --decay_step_list [50]
    >> logs/${extra_tag}/nohup.log 2>&1 &
