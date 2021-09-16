gpu=2
model=punet
extra_tag=punet_ev_smoothl1
npoint=4096
h5_file_path=nfs50_no_polarity.hdf5

if [ -d "logs/${extra_tag}" ]; then
    rm -r logs/${extra_tag}
fi

mkdir logs/${extra_tag}

nohup python -u train_smoothl1.py \
    --model ${model} \
    --batch_size 64 \
    --lr 0.002\
    --npoint ${npoint} \
    --log_dir logs/${extra_tag} \
    --gpu ${gpu} \
    --h5_file_path datas/${h5_file_path} \
    --use_decay \
    >> logs/${extra_tag}/nohup.log 2>&1 &
