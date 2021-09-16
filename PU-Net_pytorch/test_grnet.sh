gpu=3
# model=punet
model=grnet_bypass2
# extra_tag=punet_baseline
# epoch=5
extra_tag=grnet_bypass2

if [ ! -d "outputs/${extra_tag}" ]; then
    mkdir outputs/${extra_tag}
fi

for epoch in {199..199}; do

    if [ -d "outputs/${extra_tag}/${epoch}" ]; then
        rm -r outputs/${extra_tag}/${epoch}
    fi
    mkdir outputs/${extra_tag}/${epoch}
    python -u test_grnet.py \
        --save_dir outputs/${extra_tag}/${epoch}/ \
        --gpu ${gpu} \
        --model ${model} \
        --resume logs/${extra_tag}/punet_epoch_${epoch}.pth \
        --data_dir datas/test_data/nfs50_test \
        --up_ratio 1 \
        --npoint 4096;
done