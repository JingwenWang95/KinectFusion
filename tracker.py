import torch
import torch.nn as nn
from icp import ICP


class ICPTracker(nn.Module):

    def __init__(self,
                 args,
                 device,
                 ):

        super(ICPTracker, self).__init__()
        self.n_pyr = args.n_pyramids
        self.scales = list(range(self.n_pyr))
        self.n_iters = args.n_iters
        self.dampings = args.dampings
        self.construct_image_pyramids = ImagePyramids(self.scales, pool='avg')
        self.construct_depth_pyramids = ImagePyramids(self.scales, pool='max')
        self.device = device
        # initialize tracker at different levels
        self.icp_solvers = []
        for i in range(self.n_pyr):
            self.icp_solvers += [ICP(self.n_iters[i], damping=self.dampings[i])]

    @torch.no_grad()
    def forward(self, depth0, depth1, K):
        H, W = depth0.shape
        dpt0_pyr = self.construct_depth_pyramids(depth0.view(1, 1, H, W))
        dpt0_pyr = [d.squeeze() for d in dpt0_pyr]
        dpt1_pyr = self.construct_depth_pyramids(depth1.view(1, 1, H, W))
        dpt1_pyr = [d.squeeze() for d in dpt1_pyr]
        # optimization steps
        pose10 = torch.eye(4).to(self.device)  # initialize from identity
        for i in reversed(range(self.n_pyr)):
            Ki = get_scaled_K(K, i)
            pose10 = self.icp_solvers[i](pose10, dpt0_pyr[i], dpt1_pyr[i], Ki)

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


def get_scaled_K(K, l):
    if l != 0:
        Ks = K.clone()
        Ks[0, 0] /= 2 ** l
        Ks[1, 1] /= 2 ** l
        Ks[0, 2] /= 2 ** l
        Ks[1, 2] /= 2 ** l
        return Ks
    else:
        return K
