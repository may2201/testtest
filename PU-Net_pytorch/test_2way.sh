gpu=0
# model=punet
model=two_way_res
# extra_tag=punet_baseline
# epoch=5
extra_tag=two_way_res

if [ ! -d "outputs/${extra_tag}" ]; then
    mkdir outputs/${extra_tag}
fi

for epoch in {96..96}; do

    if [ -d "outputs/${extra_tag}/${epoch}" ]; then
        rm -r outputs/${extra_tag}/${epoch}
    fi
    mkdir outputs/${extra_tag}/${epoch}
    python -u test_2way.py \
        --save_dir outputs/${extra_tag}/${epoch}/ \
        --gpu ${gpu} \
        --model ${model} \
        --resume logs/${extra_tag}/punet_epoch_${epoch}.pth \
        --data_dir datas/test_data/nfs50_test \
        --up_ratio 1 \
        --npoint 4096;
done