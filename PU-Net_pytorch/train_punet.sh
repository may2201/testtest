gpu=7
model=punet
extra_tag=punet_baseline

mkdir logs/${extra_tag}

# nohup python -u train.py \
#     --model ${model} \
#     --batch_size 32 \
#     --lr 0.01\
#     --log_dir logs/${extra_tag} \
#     --gpu ${gpu} \
#     >> logs/${extra_tag}/nohup.log 2>&1 &

python train.py \
    --model ${model} \
    --batch_size 32 \
    --lr 0.01\
    --log_dir logs/${extra_tag} \
    --gpu ${gpu} \
