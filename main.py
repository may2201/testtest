from __future__ import print_function

import argparse
from os.path import exists, join, basename
from torch.utils.data import DataLoader
from data import get_training_set, get_validation_set, get_testing_set
import pandas as pd
import numpy as np
from image_folder import make_dataset
from run import EVSRCNNTrainer, EVSRCNNTester
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('E:/PycharmProjects/EVSR_3/runs')

# from dataset.data import get_training_set, get_test_set


# ===========================================================
# 训练参数设置
# ===========================================================
parser = argparse.ArgumentParser(description='eventSR')

parser.add_argument('--trainOrTest', type=str, default='Test', help='训练 batch size')
parser.add_argument('--batchSize', type=int, default=5, help='训练 batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='测试 batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='训练 epochs 数')
parser.add_argument('--lr', type=float, default=0.01, help='学习率. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='随机种子. Default=123')
parser.add_argument('--hrTrainDataStr', type=str, default='./dataset/train/LR_2')
parser.add_argument('--lrTrainDataStr', type=str, default='./dataset/train/LR_1')
parser.add_argument('--epochModelSaveStr', type=str, default='./result/epoch_%d_model')
parser.add_argument('--hrValidationDataStr', type=str, default='./dataset/validation/LR_2')
parser.add_argument('--lrValidationDataStr', type=str, default='./dataset/validation/LR_1')
parser.add_argument('--TestDataIndex', type=int, default=35)
parser.add_argument('--apsDataStr', type=str, default='E:/zihao/Winston/data/nfs-evsr-events/aps')
parser.add_argument('--apsTrainTimestampStr', type=str, default='./dataset/train/nfs_frame_timestamp')
parser.add_argument('--apsValidationTimestampStr', type=str, default='./dataset/validation/nfs_frame_timestamp')
parser.add_argument('--apsTestTimestampStr', type=str, default='./dataset/test/nfs_frame_timestamp')
parser.add_argument('--max_train_dataset_size', type=int, default=5)
parser.add_argument('--max_validation_dataset_size', type=int, default=3)
parser.add_argument('--event_norm', action='store_true')
parser.add_argument('--channel_num', default=16, type=float,
                    help="channel number of each event window.")

args = parser.parse_args()


def main():
    # ===========================================================
    # 设置train dataset & test dataset
    # ===========================================================

    if args.trainOrTest is 'Train':
        print('===> Loading datasets')
        ev_hr_paths = sorted(make_dataset(args.hrTrainDataStr, args.max_train_dataset_size))
        path_to_hr_events = ev_hr_paths[0]
        header = pd.read_csv(path_to_hr_events, delim_whitespace=True, header=None, names=['width', 'height'],
                             dtype={'width': np.int, 'height': np.int},
                             nrows=1)
        args.hr_width, args.hr_height = header.values[0]

        ev_lr_paths = sorted(make_dataset(args.lrTrainDataStr, args.max_train_dataset_size))
        path_to_lr_events = ev_lr_paths[0]
        header = pd.read_csv(path_to_lr_events, delim_whitespace=True, header=None, names=['width', 'height'],
                             dtype={'width': np.int, 'height': np.int},
                             nrows=1)
        args.lr_width, args.lr_height = header.values[0]

        train_set = get_training_set(args)
        validation_set = get_validation_set(args)
        args.train_set_size = train_set.dataset_size
        args.validation_set_size = validation_set.dataset_size
        training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
        validation_data_loader = DataLoader(dataset=validation_set, batch_size=args.testBatchSize, shuffle=False)
        model = EVSRCNNTrainer(args, training_data_loader, validation_data_loader)
        model.run()
        print('Finish!')

    elif args.trainOrTest is 'Test':
        print('===> Testing datasets')
        args.TestDataStr = './dataset/test/LR/%d.txt' % args.TestDataIndex
        path_to_test_events = args.TestDataStr
        header = pd.read_csv(path_to_test_events, delim_whitespace=True, header=None, names=['width', 'height'],
                             dtype={'width': np.int, 'height': np.int},
                             nrows=1)
        args.hr_width, args.hr_height = header.values[0]
        test_set = get_testing_set(args)
        testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
        model = EVSRCNNTester(args, testing_data_loader)
        model.run()
        print('Finish!')


if __name__ == '__main__':
    main()
