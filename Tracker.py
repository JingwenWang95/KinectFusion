import torch
import torch.nn as nn
from ICP import ICP


class ICPTracker(nn.Module):

    def __init__(self,
                 max_iter_per_pyr,
                 device,
                 n_pyr=3
                 ):

        super(ICPTracker, self).__init__()
        self.scales = [0, 1, 2, 3]
        self.construct_image_pyramids = ImagePyramids(self.scales, pool='avg')
        self.construct_depth_pyramids = ImagePyramids(self.scales, pool='max')
        self.device = device
        # initialize tracker at different levels
        self.icp0 = ICP(max_iter_per_pyr, damping=1e-6)
        self.icp1 = ICP(max_iter_per_pyr, damping=1e-4)
        self.icp2 = ICP(max_iter_per_pyr, damping=1e-4)
        self.icp3 = ICP(max_iter_per_pyr, damping=1e-2)

    @torch.no_grad()
    def forward(self, depth0, depth1, K):
        H, W = depth0.shape
        dpt0_pyr = self.construct_depth_pyramids(depth0.view(1, 1, H, W))
        dpt0_pyr = [d.squeeze() for d in dpt0_pyr]
        dpt1_pyr = self.construct_depth_pyramids(depth1.view(1, 1, H, W))
        dpt1_pyr = [d.squeeze() for d in dpt1_pyr]

        poseI = torch.eye(4).to(self.device)

        # GN update on level 3
        K3 = K >> 3
        pose3 = self.icp3(poseI, dpt0_pyr[3], dpt1_pyr[3], K3)

        # GN update on level 2
        K2 = K >> 2
        pose2 = self.icp2(pose3, dpt0_pyr[2], dpt1_pyr[2], K2)

        # GN update on level 1
        K1 = K >> 1
        pose1 = self.icp1(pose2, dpt0_pyr[1], dpt1_pyr[1], K1)

        # GN update on the raw scale
        pose10 = self.icp0(pose1, dpt0_pyr[0], dpt1_pyr[0], K)

        return pose10


class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """
    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
            x_out = [f(x).to(torch.bool) for f in self.multiscales]
        else:
            x_out = [f(x) for f in self.multiscales]
        return x_out