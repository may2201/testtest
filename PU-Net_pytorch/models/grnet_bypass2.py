import torch

from grnet_extensions.gridding import Gridding, GriddingReverse
from grnet_extensions.cubic_feature_sampling import CubicFeatureSampling

def get_model(npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False, gridding_res=64):
    return GRNet(npoint, up_ratio, use_normal, use_bn, use_res, gridding_res)

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
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(3843, 3843),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(3843, 960),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(960, 240),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Linear(240, 12)

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
            [partial_cloud, point_features_64, point_features_32, point_features_16, point_features_8], 
            dim=2
        )   # torch.Size([batch_size, 4096, 3843]) 

        # MLP
        point_features = self.fc11(point_features)  # ([batch_size, 4096, 3843]) 
        point_features = self.fc12(point_features)  # ([batch_size, 4096, 960])
        point_features = self.fc13(point_features)  # ([batch_size, 4096, 240])
        point_offset = self.fc14(point_features).view(-1, 16384, 3) # ([batch_size, 16384, 3])

        # Upsample
        dense_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 4, 1).view(-1, 16384, 3) + point_offset

        return sparse_cloud, dense_cloud

if __name__ == '__main__':
    print("hahi123")