t/test/nfs_frame_timestamp')
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
