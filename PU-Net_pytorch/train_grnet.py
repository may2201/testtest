import argparse
import os

from torch._C import dtype
from torch.optim import lr_scheduler, optimizer

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=str, default="7", help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--log_dir', default='logs/test', help='Log dir [default: logs/test_log]')
parser.add_argument('--npoint', type=int, default=1024,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epochs to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--use_decay', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.71)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=int, nargs='+', default=[30, 60])
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--h5_file_path', type=str, default="./datas/Patches_noHole_and_collected.h5")
parser.add_argument('--use_discrete', action='store_true', default=False)
parser.add_argument('--frame_loss_mode', type=int, default=0)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
# import torch.nn.functional as f
from torch.utils.data import DataLoader

from grnet_extensions.chamfer_dist import ChamferDistance
from grnet_extensions.gridding_loss import GriddingLoss
from models.grnet import GRNet

from dataset import PUNET_Dataset
import numpy as np

def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or \
       type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    train_dst = PUNET_Dataset(
        h5_file_path=args.h5_file_path, 
        npoint=args.npoint, 
        use_random=True, 
        use_norm=True, 
        split='train', 
        is_training=True, 
        is_discrete=args.use_discrete
    )
    train_loader = DataLoader(
        dataset=train_dst, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=args.workers
    )
    
    model = GRNet(
        npoint=args.npoint, 
        up_ratio=args.up_ratio, 
        use_normal=False, 
        use_bn=args.use_bn, 
        use_res=args.use_res
    )
    model.apply(init_weights)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.999)
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.decay_step_list, gamma=0.5
    )
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(    # lgtm [py/unused-local-variable]
        scales=[128], alphas=[0.1]
    )

    model.train()
    for epoch in range(args.max_epoch):
        loss_list = []
        s_cd_loss_list = []
        d_cd_loss_list = []

        for idx, batch in enumerate(train_loader):
            input_data, gt_data, _ = batch

            input_data = input_data.cuda()
            gt_data = gt_data[..., :3].contiguous().cuda()

            sparse_preds, dense_preds = model(input_data)

            sparse_loss = chamfer_dist(sparse_preds, gt_data)
            dense_loss = chamfer_dist(dense_preds, gt_data)
            loss = sparse_loss + dense_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
             
            loss_list.append(loss.item()*1000)
            s_cd_loss_list.append(sparse_loss.item()*1000)
            d_cd_loss_list.append(dense_loss.item()*1000)
            
        print(' -- epoch {}, cd loss {:.4f}, sparse cd loss {:.4f}, dense cd loss {:.4f}, lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(s_cd_loss_list), np.mean(d_cd_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))
        
        lr_scheduler.step()
        
        state = {'epoch': epoch, 'model_state': model.state_dict()}
        save_path = os.path.join(args.log_dir, 'punet_epoch_{}.pth'.format(epoch))
        torch.save(state, save_path)