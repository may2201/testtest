gpu=7
model=punet
extra_tag=punet_ev_test
h5_file_path=nfs50_no_polarity.hdf5
pth_name=punet_epoch_34.pth
npoint=4096

# for f in logs/punet_ev_test/*.pth; do
#     python eval.py \
#         --model ${model} \
#         --resume ${f} \
#         --gpu ${gpu} \
#         --npoint ${npoint} \
#         --h5_file_path datas/${h5_file_path}
# done

python eval.py \
    --model ${model} \
    --resume logs/${extra_tag}/${pth_name} \
    --gpu ${gpu} \
    --npoint ${npoint} \
    --h5_file_path datas/${h5_file_path}

python eval.py --model punet --npoint 4096 --gpu 0 --resume logs/punet_ev_l1/punet_epoch_99.pth --h5_file_path datas/nfs50_no_polarity.hdf5 --output_name punet_l1
python eval.py --model punet --npoint 4096 --gpu 1 --resume logs/punet_ev_l2/punet_epoch_99.pth --h5_file_path datas/nfs50_no_polarity.hdf5 --output_name punet_l2
python eval.py --model punet --npoint 4096 --gpu 2 --resume logs/punet_ev_smoothl1/punet_epoch_99.pth --h5_file_path datas/nfs50_no_polarity.hdf5 --output_name punet_smoothl1