gpu=7
model=punet
extra_tag=punet_baseline


if [ -d "logs/${extra_tag}" ]; then
    rm -r logs/${extra_tag}
fi

mkdir logs/${extra_tag}

nohup python -u train.py \
    --model ${model} \
    --batch_size 32 \
    --lr 0.01\
    --log_dir logs/${extra_tag} \
    --gpu ${gpu} \
    >> logs/${extra_tag}/nohup.log 2>&1 &

# python train.py \
#     --model ${model} \
#     --batch_size 64 \
#     --lr 0.02\
#     --log_dir logs/${extra_tag} \
#     --gpu ${gpu} \
