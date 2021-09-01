import argparse
import os

from torch._C import dtype

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=7, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--log_dir', default='logs/test', help='Log dir [default: logs/test_log]')
parser.add_argument('--npoint', type=int, default=1024,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epochs to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=1.0) # for repulsion loss
parser.add_argument("--beta", type=float, default=0.0) # for frame loss
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--use_decay', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.71)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[30, 60])
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--h5_file_path', type=str, default="./datas/Patches_noHole_and_collected.h5")
parser.add_argument('--use_discrete', action='store_true', default=False)
parser.add_argument('--frame_loss_mode', type=int, default=0)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader

from pointnet2 import pointnet2_utils as pn2_utils
from utils.utils import knn_point
from chamfer_distance import chamfer_distance
from auction_match import auction_match

from dataset import PUNET_Dataset
import numpy as np
import importlib


class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, nn_size=5, radius=0.07, h=0.03, eps=1e-12, frame_mode=1, frame_size=[111, 62]):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps
        self.frame_mode = frame_mode

        if frame_mode > 0:
            self.frame_size = frame_size
            sample_x, sample_y = torch.meshgrid(torch.tensor(range(frame_size[0])), torch.tensor(range(frame_size[1])))
            self.sample_points = torch.cat((torch.reshape(sample_x, (-1, 1)), torch.reshape(sample_y, (-1, 1))), dim=1)

    def get_emd_loss(self, pred, gt, pcd_radius):
        idx, _ = auction_match(pred, gt)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
        dist2 /= pcd_radius
        return torch.mean(dist2)

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = chamfer_distance(gt, pred)
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    def get_repulsion_loss(self, pred):
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        idx = idx[:, :, 1:].to(torch.int32) # remove first one
        idx = idx.contiguous() # B, N, nn

        pred = pred.transpose(1, 2).contiguous() # B, 3, N
        grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        # dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist2 = torch.maximum(dist2, (torch.zeros_like(dist2)+self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)
        # uniform_loss = torch.mean(self.radius - dist * weight) # punet
        return uniform_loss

    def get_frame_loss_1(self, pred, gt):
        pred_xy = torch.reshape(pred[..., 1:3], [-1, 2]).contiguous()
        pred_xy -= pred_xy.min(dim=0, keepdim=True).values
        pred_xy /= pred_xy.max(dim=0, keepdim=True).values
        pred_xy = pred_xy * (torch.tensor(self.frame_size)-1).cuda()

        gt_xy = torch.reshape(gt[..., 1:3], [-1, 2]).contiguous()
        gt_xy -= gt_xy.amin(dim=0, keepdim=True)
        gt_xy /= gt_xy.amax(dim=0, keepdim=True)
        gt_xy = gt_xy * (torch.tensor(self.frame_size)-1).cuda()

        coord_xy = self.sample_points.cuda().unsqueeze_(1).expand((-1, pred_xy.shape[0], -1))

        frame = lambda x, sigma: torch.sum(torch.exp(-torch.linalg.norm(x - coord_xy, dim=-1).pow(2) / sigma), dim=-1)
        pred_frame = frame(pred_xy, 0.01)
        gt_frame = frame(gt_xy, 0.01)
 
        frame_loss = f.mse_loss(pred_frame.cuda(), gt_frame.cuda())
        return frame_loss

    def forward(self, pred, gt, pcd_radius):
        cd_loss = self.get_cd_loss(pred, gt, pcd_radius) * 100
        rep_loss = self.alpha * self.get_repulsion_loss(pred)
        
        if self.frame_mode == 0:
            f_loss = torch.tensor(0.)
        elif self.frame_mode == 1:
            f_loss = self.beta * self.get_frame_loss_1(pred, gt)
        else:
            raise NotImplementedError

        return cd_loss, rep_loss, f_loss
        # return f_loss

def get_optimizer():
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.98, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    else:
        raise NotImplementedError
    
    if args.use_decay:
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in args.decay_step_list:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * args.lr_decay
            return max(cur_decay, args.lr_clip / args.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
        return optimizer, lr_scheduler
    else:
        return optimizer, None


if __name__ == '__main__':
    # train_dst = PUNET_Dataset(npoint=args.npoint, 
    #         use_random=True, use_norm=True, split='train', is_training=True)
    train_dst = PUNET_Dataset(h5_file_path=args.h5_file_path, npoint=args.npoint, 
            use_random=True, use_norm=True, split='train', is_training=True, 
            is_discrete=args.use_discrete)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, 
                        shuffle=True, pin_memory=True, num_workers=args.workers)
    # train_loader = DataLoader(train_dst, batch_size=args.batch_size, 
    #                     shuffle=True, pin_memory=True)

    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(npoint=args.npoint, up_ratio=args.up_ratio, 
                use_normal=False, use_bn=args.use_bn, use_res=args.use_res)
    model.cuda()
    
    optimizer, lr_scheduler = get_optimizer()
    loss_func = UpsampleLoss(alpha=args.alpha, beta=args.beta, frame_mode=args.frame_loss_mode)

    model.train()
    for epoch in range(args.max_epoch):
        loss_list = []
        emd_loss_list = []
        rep_loss_list = []
        f_loss_list = []
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_data, gt_data, radius_data = batch

            input_data = input_data.float().cuda()
            gt_data = gt_data.float().cuda()
            gt_data = gt_data[..., :3].contiguous()
            radius_data = radius_data.float().cuda()

            preds = model(input_data)
            emd_loss, rep_loss, f_loss = loss_func(preds, gt_data, radius_data)
            loss = emd_loss + rep_loss + f_loss
            # loss = loss_func(preds, gt_data, radius_data)

            loss.backward()
            optimizer.step()
             
            loss_list.append(loss.item())
            emd_loss_list.append(emd_loss.item())
            rep_loss_list.append(rep_loss.item())
            f_loss_list.append(f_loss.item())
            # emd_loss_list.append(0)
            # rep_loss_list.append(0)
            # f_loss_list.append(loss.item())
            
        # print(' -- epoch {}, loss {:.4f}, weighted emd loss {:.4f}, repulsion loss {:.4f}, lr {}.'.format(
        #     epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(rep_loss_list), \
        #     optimizer.state_dict()['param_groups'][0]['lr']))
        print(' -- epoch {}, loss {:.4f}, weighted cd loss {:.4f}, repulsion loss {:.4f}, frame loss {:.4f}, lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(rep_loss_list), np.mean(f_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        # if (epoch + 1) % 5 == 0:
        state = {'epoch': epoch, 'model_state': model.state_dict()}
        save_path = os.path.join(args.log_dir, 'punet_epoch_{}.pth'.format(epoch))
        torch.save(state, save_path)