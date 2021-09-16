import torch
from torchviz import make_dot
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str("7")

from torch._C import dtype

import torch.nn as nn
import torch.nn.functional as f
# from torch.utils.data import DataLoader

from pointnet2 import pointnet2_utils as pn2_utils
from utils.utils import knn_point
from chamfer_distance import chamfer_distance
# from auction_match import auction_match

# from dataset import PUNET_Dataset
# import numpy as np
# import importlib

from torch.autograd import gradcheck

class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, nn_size=5, radius=0.07, h=0.03, eps=1e-12, frame_mode=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps
        self.frame_mode = frame_mode

    # def get_emd_loss(self, pred, gt, pcd_radius):
    #     idx, _ = auction_match(pred, gt)
    #     matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    #     matched_out = matched_out.transpose(1, 2).contiguous()
    #     dist2 = (pred - matched_out) ** 2
    #     dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
    #     dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
    #     dist2 /= pcd_radius
    #     return torch.mean(dist2)

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = chamfer_distance(gt.to(torch.float), pred.to(torch.float))
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    # def get_repulsion_loss(self, pred):
    #     _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
    #     idx = idx[:, :, 1:].to(torch.int32) # remove first one
    #     idx = idx.contiguous() # B, N, nn

    #     pred = pred.transpose(1, 2).contiguous() # B, 3, N
    #     grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

    #     grouped_points = grouped_points - pred.unsqueeze(-1)
    #     dist2 = torch.sum(grouped_points ** 2, dim=1)
    #     # dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
    #     dist2 = torch.maximum(dist2, (torch.zeros_like(dist2)+self.eps).cuda())
    #     dist = torch.sqrt(dist2)
    #     weight = torch.exp(- dist2 / self.h ** 2)

    #     uniform_loss = torch.mean((self.radius - dist) * weight)
    #     # uniform_loss = torch.mean(self.radius - dist * weight) # punet
    #     return uniform_loss

    def get_frame_loss_1(self, pred, gt, frame_size=[222, 124]):
        pred_xy = torch.reshape(pred[..., 1:3], [-1, 2]).contiguous()
        # pred_xy = pred[..., 1:3]
        pred_xy -= pred_xy.min(dim=0, keepdim=True).values
        pred_xy /= pred_xy.max(dim=0, keepdim=True).values

        gt_xy = torch.reshape(gt[..., 1:3], [-1, 2]).contiguous()
        # gt_xy = gt[..., 1:3]
        gt_xy -= gt_xy.min(dim=0, keepdim=True).values
        gt_xy /= gt_xy.max(dim=0, keepdim=True).values

        # pred_xy = torch.round(pred_xy * (torch.tensor(frame_size)-1).cuda()).to(torch.int64)
        # gt_xy = torch.round(gt_xy * (torch.tensor(frame_size)-1).cuda()).to(torch.int64)
        # pred_frame = torch.zeros(frame_size).cuda().index_put((pred_xy[..., 0], pred_xy[..., 1]), values=torch.tensor(1.).cuda(), accumulate=True)
        # gt_frame = torch.zeros(frame_size).cuda().index_put((gt_xy[..., 0], gt_xy[..., 1]), values=torch.tensor(1.).cuda(), accumulate=True)
        ####  slow
        # pred_frame = torch.zeros(frame_size).cuda()
        # for ii in range(pred_xy.shape[0]):
        #     pred_frame[pred_xy[ii, 0], pred_xy[ii, 1]] += 1.
        # gt_frame = torch.zeros(frame_size).cuda()
        # for ii in range(gt.shape[0]):
        #     gt_frame[gt_xy[ii, 0], gt_xy[ii, 1]] += 1.
        ####  slow

        gt_xy = torch.round(gt_xy.cpu() * (torch.tensor(frame_size)-1)).to(torch.int64)
        pred_xy = torch.round(pred_xy.cpu() * (torch.tensor(frame_size)-1)).to(torch.int64)
        pred_frame = torch.zeros(frame_size, requires_grad=True).index_put_((pred_xy[..., 0], pred_xy[..., 1]), values=torch.tensor(1.), accumulate=True)
        gt_frame = torch.zeros(frame_size, requires_grad=True).index_put_((gt_xy[..., 0], gt_xy[..., 1]), values=torch.tensor(1.), accumulate=True)


        frame_loss = f.mse_loss(pred_frame.cuda(), gt_frame.cuda(), reduction='mean')

        return frame_loss

    def forward(self, pred, gt, pcd_radius):
        cd_loss = self.get_cd_loss(pred, gt, pcd_radius) * 100
        # rep_loss = self.alpha * self.get_repulsion_loss(pred)
        
        if self.frame_mode == 0:
            f_loss = torch.tensor(0.)
        elif self.frame_mode == 1:
            f_loss = self.beta * self.get_frame_loss_1(pred, gt)
        elif self.frame_mode == 2:
            f_loss = self.beta * self.get_frame_loss_2(pred, gt)   
        else:
            raise NotImplementedError

        return cd_loss + f_loss


if __name__ == '__main__':
    loss = UpsampleLoss(alpha=1, frame_mode=1)
    loss = loss.double()
    pred = torch.randn([8,16384,3], requires_grad=True, dtype=torch.double).cuda()
    gt = torch.randn([8,16384,3], requires_grad=True, dtype=torch.double).cuda()
    losss = loss(pred, gt, 1)
    print("raw losss: {}".format(losss))

    loss = UpsampleLoss(alpha=1, frame_mode=1)
    loss = loss.double()
    pred = torch.randn([8,16384,3], requires_grad=True, dtype=torch.double).cuda()
    gt = torch.randn([8,16384,3], requires_grad=True, dtype=torch.double).cuda()
    oo = gradcheck(loss, (pred, gt, 1))
    print("gradcheck: {}".format(gradcheck))