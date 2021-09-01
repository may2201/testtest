gpu=5
model=punet
extra_tag=punet_ev_frame001_norep_ur1
npoint=4096
h5_file_path=nfs50/nfs50_no_polarity.hdf5

if [ -d "logs/${extra_tag}" ]; then
    rm -r logs/${extra_tag}
fi

mkdir logs/${extra_tag}

nohup python -u train.py \
    --model ${model} \
    --batch_size 32 \
    --lr 0.001\
    --npoint ${npoint} \
    --log_dir logs/${extra_tag} \
    --gpu ${gpu} \
    --h5_file_path datas/${h5_file_path} \
    --beta 1.0 \
    --alpha 0.0 \
    --up_ratio 1 \
    --frame_loss_mode 1 \
    >> logs/${extra_tag}/nohup.log 2>&1 &
