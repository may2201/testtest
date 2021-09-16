import torch
import torch.nn as nn

from pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
import pointnet2.pytorch_utils as pt_utils

from grnet_extensions.gridding import Gridding, GriddingReverse
from grnet_extensions.cubic_feature_sampling import CubicFeatureSampling

def get_model(npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False, gridding_res=64):
    return fengheNet(npoint, up_ratio, use_normal, use_bn, use_res, gridding_res)


class PUNet(nn.Module):
    def __init__(self, npoint=4096, up_ratio=1, use_normal=False, use_bn=False, use_res=False):
        super().__init__()

        self.npoint = 4096
        self.use_normal = use_normal
        self.up_ratio = 1

        self.npoints = [
            npoint, 
            npoint // 2, 
            npoint // 4, 
            npoint // 8
        ]

        mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]
        nsamples = [32, 32, 32, 32]

        ## for 4 downsample layers
        in_ch = 0 if not use_normal else 3
        self.SA_modules = nn.ModuleList()
        for k in range(len(self.npoints)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=[in_ch] + mlps[k],
                    use_xyz=True,
                    use_res=use_res,
                    bn=use_bn
                )
            )
            in_ch = mlps[k][-1]

        ## upsamples for layer 2 ~ 4
        self.FP_Modules = nn.ModuleList()
        for k in range(len(self.npoints) - 1):
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=[mlps[k + 1][-1], 512], 
                    bn=use_bn
                )
            )

    def forward(self, points, npoint=None):
        if npoint is None:
            npoints = [None] * len(self.npoints)
        else:
            npoints = []
            for k in range(len(self.npoints)):
                npoints.append(npoint // 2 ** k)

        ## points: bs, N, 3/6
        xyz = points[..., :3].contiguous()
        feats = points[..., 3:].transpose(1, 2).contiguous() \
            if self.use_normal else None

        ## downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_xyz, lk_feats = self.SA_modules[k](l_xyz[k], l_feats[k], npoint=npoints[k])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        ## upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            upk_feats = self.FP_Modules[k](xyz, l_xyz[k + 2], None, l_feats[k + 2])
            up_feats.append(upk_feats)

        ## aggregation
        # [xyz, l0, l1, l2, l3]
        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_feats[1],
            *up_feats], dim=1).unsqueeze(-1) # bs, mid_ch, N, 1
        
        return feats.squeeze(-1).transpose(1, 2).contiguous() # bs, N, mid_ch




class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()

class GRNet(torch.nn.Module):
    # def __init__(self, cfg):
    def __init__(self, npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False, gridding_res=64):
        super().__init__()
        super(GRNet, self).__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio
        self.gridding_res = gridding_res

        self.gridding = Gridding(scale=128)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(32768, 4096),
            torch.nn.ReLU()
        )

        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(4096, 32768),
            torch.nn.ReLU()
        )
        self.dconv_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=128)
        self.point_sampling = RandomPointSampling(n_points=4096)
        self.feature_sampling = CubicFeatureSampling()

    def forward(self, points, npoint=None):
        partial_cloud = points[..., :3].contiguous()
        # print(partial_cloud.size())     # torch.Size([batch_size, 4096, 3])

        # Gridding
        pt_features_128_l = self.gridding(partial_cloud).view(-1, 1, 128, 128, 128)
        # print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 128, 128, 128])

        # 3D CNN
        pt_features_64_l = self.conv1(pt_features_128_l)    # ([batch_size, 32, 64, 64, 64])
        pt_features_32_l = self.conv2(pt_features_64_l)     # ([batch_size, 64, 32, 32, 32])
        pt_features_16_l = self.conv3(pt_features_32_l)     # ([batch_size, 128, 16, 16, 16])
        pt_features_8_l = self.conv4(pt_features_16_l)      # ([batch_size, 256, 8, 8, 8])
        pt_features_4_l = self.conv_1(pt_features_8_l)       # ([batch_size, 512, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 32768)) # ([batch_size, 4096])
        pt_features_4_r = self.fc6(features).view(-1, 512, 4, 4, 4) + pt_features_4_l # ([batch_size, 512, 4, 4, 4])
        pt_features_8_r = self.dconv_2(pt_features_4_r) + pt_features_8_l   # ([batch_size, 256, 8, 8, 8])
        pt_features_16_r = self.dconv7(pt_features_8_r) + pt_features_16_l  # ([batch_size, 128, 16, 16, 16]) 
        pt_features_32_r = self.dconv8(pt_features_16_r) + pt_features_32_l # ([batch_size, 64, 32, 32, 32])
        pt_features_64_r = self.dconv9(pt_features_32_r) + pt_features_64_l # ([batch_size, 32, 64, 64, 64])
        pt_features_128_r = self.dconv10(pt_features_64_r) + pt_features_128_l # ([batch_size, 1, 128, 128, 128])

        # Gridding Reverse
        sparse_cloud = self.gridding_rev(pt_features_128_r.squeeze(dim=1)) # ([batch_size, 2097152, 3])

        # Cubic Feature Sampling
        sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud) # ([batch_size, 4096, 3])
        
        point_features_64 = self.feature_sampling(
            sparse_cloud, pt_features_64_r).view(-1, 4096, 256) # ([batch_size, 4096, 256])
        point_features_32 = self.feature_sampling(
            sparse_cloud, pt_features_32_r).view(-1, 4096, 512) # ([batch_size, 4096, 512])
        point_features_16 = self.feature_sampling(
            sparse_cloud, pt_features_16_r).view(-1, 4096, 1024) # ([batch_size, 4096, 1024])
        point_features_8 = self.feature_sampling(
            sparse_cloud, pt_features_8_r).view(-1, 4096, 2048) # ([batch_size, 4096, 2048])

        point_features = torch.cat(
            [point_features_64, point_features_32, point_features_16, point_features_8], 
            dim=2
        )   # torch.Size([batch_size, 4096, 3840]) 

        return point_features


class fengheNet(torch.nn.Module):
    # def __init__(self, cfg):
    def __init__(self, npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False, gridding_res=64):
        super().__init__()
        
        # bs, N, 1603
        self.punet = PUNet(npoint=4096, up_ratio=1, use_normal=use_normal, use_bn=use_bn, use_res=use_res) 
        # bs, N, 3840
        self.grnet = GRNet(npoint=4096, up_ratio=1, use_normal=use_normal, use_bn=use_bn, use_res=use_res, gridding_res=gridding_res)

        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(5443, 5443),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(5443, 2560),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(2560, 1280),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Sequential(
            torch.nn.Linear(1280, 640),
            torch.nn.ReLU()
        )
        self.fc15 = torch.nn.Sequential(
            torch.nn.Linear(640, 240),
            torch.nn.ReLU()
        )
        self.fc16 = torch.nn.Linear(240, 12)

    def forward(self, points, npoint=None):
        
        pu_features = self.punet(points)
        gr_features = self.grnet(points)

        point_features = torch.cat([pu_features, gr_features], dim=2)   # torch.Size([batch_size, 4096, 5443]) 

        # MLP
        point_features = self.fc11(point_features)  # ([batch_size, 4096, 5443]) 
        point_features = self.fc12(point_features)  # ([batch_size, 4096, 2560])
        point_features = self.fc13(point_features)  # ([batch_size, 4096, 1280])
        point_features = self.fc14(point_features)  # ([batch_size, 4096, 640])
        point_features = self.fc15(point_features)  # ([batch_size, 4096, 240])
        point_offset = self.fc16(point_features).view(-1, 16384, 3) # ([batch_size, 16384, 3])

        point = points.unsqueeze(dim=2).repeat(1, 1, 4, 1).view(-1, 16384, 3) + point_offset


        return point

if __name__ == '__main__':
    punet = PUNet(npoint=4096, up_ratio=1)
    points = torch.rand((16,4096,3))
    o = punet(points)
    print(o.shape)