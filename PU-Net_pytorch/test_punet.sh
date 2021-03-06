gpu=0
# model=punet
model=punet
# extra_tag=punet_baseline
# epoch=5
extra_tag=punet_ev_frame_lr0001_b1_norep

if [ ! -d "outputs/${extra_tag}" ]; then
    mkdir outputs/${extra_tag}
fi

for epoch in {99..99}; do

    if [ -d "outputs/${extra_tag}/${epoch}" ]; then
        rm -r outputs/${extra_tag}/${epoch}
    fi
    mkdir outputs/${extra_tag}/${epoch}
    python -u test.py \
        --model ${model} \
        --save_dir outputs/${extra_tag}/${epoch}/ \
        --gpu ${gpu} \
        --resume logs/${extra_tag}/punet_epoch_${epoch}.pth \
        --data_dir datas/test_data/nfs50_test \
        --up_ratio 1 \
        --npoint 4096 ;
done