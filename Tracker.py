import torch
import torch.nn as nn

from models.algorithms import ImagePyramids
from ICP import ICP, ColorICP


class ICPTracker(nn.Module):

    def __init__(self,
                 max_iter_per_pyr,
                 device,
                 direction='forward',
                 timers=None):

        super(ICPTracker, self).__init__()
        self.scales = [0, 1, 2, 3]
        self.construct_image_pyramids = ImagePyramids(self.scales, pool='avg')
        self.construct_depth_pyramids = ImagePyramids(self.scales, pool='max')
        self.device = device
        self.timers = timers
        self.direction = direction
        # initialize tracker at different levels
        self.tr_update0 = ICP(max_iter_per_pyr, damping=1e-6, direction=direction, timers=timers)
        self.tr_update1 = ICP(max_iter_per_pyr, damping=1e-4, direction=direction, timers=timers)
        self.tr_update2 = ICP(max_iter_per_pyr, damping=1e-4, direction=direction, timers=timers)
        self.tr_update3 = ICP(max_iter_per_pyr, damping=1e-2, direction=direction, timers=timers)

    def forward(self, depth0, depth1, K, dpt1_pyr=None, vtx1_pyr=None, nrm1_pyr=None, msk1_pyr=None):
        H, W = depth0.shape
        dpt0_pyr = self.construct_depth_pyramids(depth0.view(1, 1, H, W))
        dpt0_pyr = [d.squeeze() for d in dpt0_pyr]

        # TODO: this function only accepts batched inputs
        if dpt1_pyr is None:
            dpt1_pyr = self.construct_depth_pyramids(depth1.view(1, 1, H, W))
            dpt1_pyr = [d.squeeze() for d in dpt1_pyr]

        if vtx1_pyr is None:
            vtx1_pyr = [None] * 4

        # For rendered normals using avg pooling doesn't make sense
        if nrm1_pyr is None:
            nrm1_pyr = [None] * 4

        if msk1_pyr is None:
            msk1_pyr = [None] * 4

        poseI = torch.eye(4).to(self.device)

        if self.timers: self.timers.tic('trust-region update')
        # trust region update on level 3
        K3 = K >> 3
        pose3 = self.tr_update3(poseI, dpt0_pyr[3], dpt1_pyr[3], K3, vertex1=vtx1_pyr[3], normal1=nrm1_pyr[3], mask1=msk1_pyr[3])

        # trust region update on level 2
        K2 = K >> 2
        pose2 = self.tr_update2(pose3, dpt0_pyr[2], dpt1_pyr[2], K2, vertex1=vtx1_pyr[2], normal1=nrm1_pyr[2], mask1=msk1_pyr[2])

        # trust region update on level 1
        K1 = K >> 1
        pose1 = self.tr_update1(pose2, dpt0_pyr[1], dpt1_pyr[1], K1, vertex1=vtx1_pyr[1], normal1=nrm1_pyr[1], mask1=msk1_pyr[1])

        # trust-region update on the raw scale
        pose0 = self.tr_update0(pose1, dpt0_pyr[0], dpt1_pyr[0], K, vertex1=vtx1_pyr[0], normal1=nrm1_pyr[0], mask1=msk1_pyr[0])

        if self.timers: self.timers.toc('trust-region update')

        return pose0


class ColorICPTracker(nn.Module):

    def __init__(self,
                 max_iter_per_pyr,
                 device,
                 direction='backward',
                 timers=None):

        super(ColorICPTracker, self).__init__()
        self.scales = [0, 1, 2, 3]
        self.construct_image_pyramids = ImagePyramids(self.scales, pool='avg')
        self.construct_depth_pyramids = ImagePyramids(self.scales, pool='max')
        self.device = device
        self.timers = timers
        self.direction = direction
        # initialize tracker at different levels
        self.tr_update0 = ColorICP(max_iter_per_pyr, damping=1e-6, direction=direction, timers=timers)
        self.tr_update1 = ColorICP(max_iter_per_pyr, damping=1e-4, direction=direction, timers=timers)
        self.tr_update2 = ColorICP(max_iter_per_pyr, damping=1e-4, direction=direction, timers=timers)
        self.tr_update3 = ColorICP(2 * max_iter_per_pyr, damping=1e-4, direction=direction, timers=timers)

    def forward(self, depth0, depth1, color0, color1, K, dpt1_pyr=None, vtx1_pyr=None, nrm1_pyr=None, msk1_pyr=None):
        H, W = depth0.shape
        rgb0_pyr = self.construct_image_pyramids(color0.permute(2, 0, 1).view(1, 3, H, W))
        rgb0_pyr = [r.squeeze().permute(1, 2, 0) for r in rgb0_pyr]
        rgb1_pyr = self.construct_image_pyramids(color1.permute(2, 0, 1).view(1, 3, H, W))
        rgb1_pyr = [r.squeeze().permute(1, 2, 0) for r in rgb1_pyr]
        dpt0_pyr = self.construct_depth_pyramids(depth0.view(1, 1, H, W))
        dpt0_pyr = [d.squeeze() for d in dpt0_pyr]

        # TODO: this function only accepts batched inputs
        if dpt1_pyr is None:
            dpt1_pyr = self.construct_depth_pyramids(depth1.view(1, 1, H, W))
            dpt1_pyr = [d.squeeze() for d in dpt1_pyr]

        if vtx1_pyr is None:
            vtx1_pyr = [None] * 4

        # For rendered normals using avg pooling doesn't make sense
        if nrm1_pyr is None:
            nrm1_pyr = [None] * 4

        if msk1_pyr is None:
            msk1_pyr = [None] * 4

        poseI = torch.eye(4).to(self.device)

        if self.timers: self.timers.tic('trust-region update')
        # trust region update on level 3
        K3 = K >> 3
        pose3 = self.tr_update3(poseI, dpt0_pyr[3], dpt1_pyr[3], rgb0_pyr[3], rgb1_pyr[3], K3, vertex1=vtx1_pyr[3], normal1=nrm1_pyr[3], mask1=msk1_pyr[3])

        # trust region update on level 2
        K2 = K >> 2
        pose2 = self.tr_update2(pose3, dpt0_pyr[2], dpt1_pyr[2], rgb0_pyr[2], rgb1_pyr[2], K2, vertex1=vtx1_pyr[2], normal1=nrm1_pyr[2], mask1=msk1_pyr[2])

        # trust region update on level 1
        K1 = K >> 1
        pose1 = self.tr_update1(pose2, dpt0_pyr[1], dpt1_pyr[1], rgb0_pyr[1], rgb1_pyr[1], K1, vertex1=vtx1_pyr[1], normal1=nrm1_pyr[1], mask1=msk1_pyr[1])

        # trust-region update on the raw scale
        pose0 = self.tr_update0(pose1, dpt0_pyr[0], dpt1_pyr[0], rgb0_pyr[0], rgb1_pyr[0], K, vertex1=vtx1_pyr[0], normal1=nrm1_pyr[0], mask1=msk1_pyr[0])

        if self.timers: self.timers.toc('trust-region update')

        return pose0

