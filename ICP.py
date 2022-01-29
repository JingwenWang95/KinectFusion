import torch
import torch.nn as nn
import torch.nn.functional as F

import models.geometry as geometry
import cv2
import numpy as np


# forward ICP
class ICP(nn.Module):
    def __init__(self,
                 max_iter=3,
                 damping=1e-3,
                 direction="forward",
                 timers=None):
        """
        :param max_iter, maximum number of iterations
        :param timers, if yes, counting time for each step
        """
        super(ICP, self).__init__()

        self.max_iterations = max_iter
        self.damping = damping
        assert direction in ["forward", "backward"]
        self.direction = direction
        self.timers = timers

    @torch.no_grad()
    def forward(self, pose10, depth0, depth1, K, vertex1=None, normal1=None, mask1=None):
        """
        :param pose10:
        :param vertex0:
        :param normal0:
        :param depth1:
        :param K:
        :return:
        """

        # create vertex and normal for current frame
        vertex0 = compute_vertex(depth0, K)
        normal0 = compute_normal(vertex0)
        mask0 = depth0 > 0.

        # TODO: vertex0 and normal0 can be rendered directly
        if vertex1 is None:
            vertex1 = compute_vertex(depth1, K)
        if normal1 is None:
            normal1 = compute_normal(vertex1)
        if mask1 is None:
            mask1 = depth1 > 0.

        #  # visualize surface normal image.
        # img = (((normal1.permute(0,2,3,1)[0,:,:,:]+1.0)*128.0).cpu().numpy()).astype('uint8')
        # cv2.imshow('normal', img)
        # cv2.waitKey(0)

        for idx in range(self.max_iterations):
            # compute residuals
            if self.timers: self.timers.tic('compute warping residuals')
            residuals, J_F_p, occ = self.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, mask0, pose10, K)
            if self.timers: self.timers.toc('compute warping residuals')

            if self.timers: self.timers.tic('pre-compute JtWJ')
            JtWJ = self.compute_jtj(J_F_p)  # [B, 6, 6]
            JtR = self.compute_jtr(J_F_p, residuals)
            if self.timers: self.timers.toc('pre-compute JtWJ')

            if self.timers: self.timers.tic('solve x=A^{-1}b')
            pose10 = self.GN_solver(JtWJ, JtR, pose10, damping=self.damping)
            if self.timers: self.timers.toc('solve x=A^{-1}b')

        # weights = torch.ones(residuals.shape).type_as(residuals)
        # print('---')
        return pose10

    def compute_residuals_jacobian(self, vertex0, vertex1, normal0, normal1, mask0, pose10, K):
        R = pose10[:3, :3]
        t = pose10[:3, 3]
        H, W, C = vertex0.shape

        rot_vertex0_to1 = (R @ vertex0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)
        vertex0_to1 = rot_vertex0_to1 + t[None, None, :]
        normal0_to1 = (R @ normal0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_, y_, z_ = vertex0_to1[..., 0], vertex0_to1[..., 1], vertex0_to1[..., 2]  # [h, w]
        u_ = (x_ / z_) * fx + cx  # [h, w]
        v_ = (y_ / z_) * fy + cy  # [h, w]

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)
        # TODO: interpolation doesn't make sense for rendered vertex and normals!!!
        # data association by warping
        r_vertex1 = warp_features(vertex1, u_, v_)  # [h, w, 3]
        r_normal1 = warp_features(normal1, u_, v_)  # [h, w, 3]
        mask1 = r_vertex1[..., -1] > 0.

        diff = vertex0_to1 - r_vertex1  # [h, w, 3]
        normal_diff = (normal0_to1 * r_normal1).sum(dim=-1)

        # occlusion
        occ = ~inviews | (diff.norm(p=2, dim=-1) > 0.10)
              #| (normal_diff < 0.7)   # since normal is estimated from the noise depth, might not be very useful

        # point-to-plane residuals
        res = (r_normal1 * diff).sum(dim=-1)  # [h, w]
        # point-to-plane  jacobians
        J_trs = r_normal1.view(-1, 3)  # [hw, 3]
        J_rot = -torch.bmm(J_trs.unsqueeze(dim=1), batch_skew(vertex0_to1.view(-1, 3))).squeeze()   # [hw, 3]

        # # point-to-point residuals
        # res = diff.view(B,3,H,W)  # [B,3,HXW]
        # # point-to-point jacobians
        # J_trs = torch.eye(3).type_as(res).view(1,1,3,3).expand(B,H*W,3,3)
        # J_rot = -geometry.batch_skew(rot_vertex0_to1.permute(0,2,1).contiguous().view(-1,3))
        # J_rot = J_rot.view(B, H*W, 3, 3)
        # # compose jacobians
        # J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B, H*W, 3, 6]
        # J_F_p = J_F_p.permute(0, 2, 1, 3)  # [B, 3, HXW, 6]
        # occ = occ.expand(B,3,H,W)

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1).view(H, W, 6)  # follow the order of [rot, trs]  [hw, 1, 6]

        # covariance-normalized
        # dpt0 = vertex0[..., -1]
        # sigma_icp = self.compute_sigma(dpt_l=dpt0, normal_r=r_normal1, rot=R)  # [h, w, 1]
        # res = res / (sigma_icp + 1e-8)  # [h, w, 1]
        # J_F_p = J_F_p / (sigma_icp + 1e-8)  # [h, w, 6]

        J_F_p[occ | ~mask0 | ~mask1] = 0.
        res[occ | ~mask0 | ~mask1] = 0.
        res = res.view(-1, 1)  # [hw, 1]
        # res, w = get_robust_res(res, 0.05)
        # follow the conversion of negating the jacobian here
        if self.direction == "forward":
            J_F_p = -J_F_p.view(-1, 1, 6)
        elif self.direction == "backward":
            J_F_p = J_F_p.view(-1, 1, 6)
        # J_F_p *= w[..., None]

        return res, J_F_p, occ

    def compute_sigma(self, dpt_l, normal_r, rot, dataset='TUM'):
        # obtain sigma (hardcoded)
        if dataset == 'TUM':
            sigma_disp = 0.4  # 5.5
            sigma_xy = 5.5  # 5.5
            baseline = 1.0  # 0.075
            focal = 525.0
        else:
            raise NotImplementedError()

        H, W, C = normal_r.shape

        # compute sigma on depth using stereo model
        sigma_depth = torch.empty(H, W, 3).type_as(dpt_l)
        sigma_depth[..., 0:2] = dpt_l[..., None] / focal * sigma_xy
        sigma_depth[..., -1] = dpt_l * dpt_l * sigma_disp / (focal * baseline)

        J = (normal_r.view(-1, 3) @ rot).view(H, W, 3)  # [h, w, 3]
        cov_icp = (J * sigma_depth * sigma_depth * J).sum(dim=-1, keepdim=True)  # [h, w ,1]

        sigma_icp = torch.sqrt(cov_icp + 1e-8)
        return sigma_icp

    def compute_jtj(self, jac):
        # J in the dimension of (HW, C, 6)
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtj = torch.bmm(jacT, jac).sum(0)  # [6, 6]
        return jtj  # [6, 6]

    def compute_jtr(self, jac, res):
        # J in the dimension of (HW, C, 6)
        # res in the dimension of [HW, C]
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtr = torch.bmm(jacT, res.unsqueeze(-1)).sum(0)  # [6, 1]
        return jtr  # [6, 1]

    def GN_solver(self, JtJ, JtR, pose0, damping=1e-6):

        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ, damping)
        # Hessian = JtJ
        if self.direction == "forward":
            updated_pose = forward_update_pose(Hessian, JtR, pose0)
        else:
            updated_pose = inverse_update_pose(Hessian, JtR, pose0)

        return updated_pose


# backward ICP + photometric
class ColorICP(nn.Module):
    def __init__(self,
                 max_iter=3,
                 damping=1e-3,
                 direction="backward",
                 timers=None):
        """
        :param max_iter, maximum number of iterations
        :param timers, if yes, counting time for each step
        """
        super(ColorICP, self).__init__()

        self.max_iterations = max_iter
        self.damping = damping
        assert direction in ["forward", "backward"]
        self.direction = direction
        self.timers = timers

    @torch.no_grad()
    def forward(self, pose10, depth0, depth1, x0, x1, K, vertex1=None, normal1=None, mask1=None):
        """
        :param pose10:
        :param vertex0:
        :param normal0:
        :param depth1:
        :param K:
        :return:
        """

        H, W, C = x0.shape
        # create vertex and normal for current frame
        vertex0 = compute_vertex(depth0, K)
        normal0 = compute_normal(vertex0)
        mask0 = depth0 > 0.

        # TODO: vertex0 and normal0 can be rendered directly
        if vertex1 is None:
            vertex1 = compute_vertex(depth1, K)
        if normal1 is None:
            normal1 = compute_normal(vertex1)
        if mask1 is None:
            mask1 = depth1 > 0.

        if self.timers: self.timers.tic('compute pre-computable Jacobian components')
        J_rgb_pre = self.precompute_jacobian(depth0, x0, K)  # [h, w, c, 6]
        if self.timers: self.timers.toc('compute pre-computable Jacobian components')

        for idx in range(self.max_iterations):
            # compute residuals
            if self.timers: self.timers.tic('compute warping residuals')
            residuals_icp, J_icp, invalid_mask = self.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, mask0, pose10, K)
            if self.timers: self.timers.toc('compute warping residuals')

            if self.timers: self.timers.tic('compute icp JtWJ')
            JtWJ_icp = self.compute_jtj(J_icp)  # [B, 6, 6]
            JtR_icp = self.compute_jtr(J_icp, residuals_icp)
            if self.timers: self.timers.toc('compute icp JtWJ')

            residuals_rgb, invalid_mask = self.compute_inverse_residuals(pose10, vertex0, vertex1, x0, x1, K)
            residuals_rgb = residuals_rgb.view(-1, C)
            J_rgb = J_rgb_pre.clone()
            J_rgb[invalid_mask, :, :] = 0.
            J_rgb = J_rgb.view(-1, C, 6)
            # residuals_rgb, w = get_robust_res(residuals_rgb, 0.25)
            # J_rgb *= w[..., None]
            JtWJ_rgb = self.compute_jtj(J_rgb)
            JtR_rgb = self.compute_jtr(J_rgb, residuals_rgb)

            lambda_icp, lambda_rgb = 1., 1e-6
            JtWJ = lambda_icp * JtWJ_icp + lambda_rgb * JtWJ_rgb
            JtR = lambda_icp * JtR_icp + lambda_rgb * JtR_rgb

            if self.timers: self.timers.tic('solve x=A^{-1}b')
            pose10 = self.GN_solver(JtWJ, JtR, pose10, damping=self.damping)
            if self.timers: self.timers.toc('solve x=A^{-1}b')

        # weights = torch.ones(residuals.shape).type_as(residuals)
        # print('---')
        return pose10

    def compute_residuals_jacobian(self, vertex0, vertex1, normal0, normal1, mask0, pose10, K):
        """
        :param vertex0: template vertex map (live image)
        :param vertex1: image vertex map (last frame)
        :param normal0:
        :param normal1:
        :param mask0:
        :param pose10:
        :param K:
        :return:
        """
        R = pose10[:3, :3]
        t = pose10[:3, 3]
        H, W, C = vertex0.shape

        rot_vertex0_to1 = (R @ vertex0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)
        vertex0_to1 = rot_vertex0_to1 + t[None, None, :]
        normal0_to1 = (R @ normal0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_, y_, z_ = vertex0_to1[..., 0], vertex0_to1[..., 1], vertex0_to1[..., 2]  # [h, w]
        u_ = (x_ / z_) * fx + cx  # [h, w]
        v_ = (y_ / z_) * fy + cy  # [h, w]

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)
        # TODO: interpolation doesn't make sense for rendered vertex and normals!!!
        # data association by warping
        r_vertex1 = warp_features(vertex1, u_, v_)  # [h, w, 3]
        r_normal1 = warp_features(normal1, u_, v_)  # [h, w, 3]
        mask1 = r_vertex1[..., -1] > 0.

        diff = vertex0_to1 - r_vertex1  # [h, w, 3]
        normal_diff = (normal0_to1 * r_normal1).sum(dim=-1)

        # occlusion
        occ = ~inviews | (diff.norm(p=2, dim=-1) > 0.10)
        #| (normal_diff < 0.7)   # since normal is estimated from the noise depth, might not be very useful

        # point-to-plane residuals
        res = (r_normal1 * diff).sum(dim=-1)  # [h, w]
        # point-to-plane  jacobians
        J_trs = r_normal1.view(-1, 3)  # [hw, 3]
        J_rot = -torch.bmm(J_trs.unsqueeze(dim=1), batch_skew(vertex0_to1.view(-1, 3))).squeeze()   # [hw, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1).view(H, W, 6)  # follow the order of [rot, trs]  [hw, 1, 6]

        # covariance-normalized
        # dpt0 = vertex0[..., -1]
        # sigma_icp = self.compute_sigma(dpt_l=dpt0, normal_r=r_normal1, rot=R)  # [h, w, 1]
        # res = res / (sigma_icp.squeeze() + 1e-8)  # [h, w, 1]
        # J_F_p = J_F_p / (sigma_icp + 1e-8)  # [h, w, 6]

        invalid_mask = occ | ~mask0 | ~mask1
        J_F_p[invalid_mask] = 0.
        res[invalid_mask] = 0.
        res = res.view(-1, 1)  # [hw, 1]
        res, w = get_robust_res(res, 0.05)
        # follow the conversion of negating the jacobian here
        if self.direction == "forward":
            J_F_p = -J_F_p.view(-1, 1, 6)
        elif self.direction == "backward":
            J_F_p = J_F_p.view(-1, 1, 6)
        J_F_p *= w[..., None]

        return res, J_F_p, invalid_mask

    def compute_sigma(self, dpt_l, normal_r, rot, dataset='TUM'):
        # obtain sigma (hardcoded)
        if dataset == 'TUM':
            sigma_disp = 0.4  # 5.5
            sigma_xy = 5.5  # 5.5
            baseline = 1.0  # 0.075
            focal = 525.0
        else:
            raise NotImplementedError()

        H, W, C = normal_r.shape

        # compute sigma on depth using stereo model
        sigma_depth = torch.empty(H, W, 3).type_as(dpt_l)
        sigma_depth[..., 0:2] = dpt_l[..., None] / focal * sigma_xy
        sigma_depth[..., -1] = dpt_l * dpt_l * sigma_disp / (focal * baseline)

        J = (normal_r.view(-1, 3) @ rot).view(H, W, 3)  # [h, w, 3]
        cov_icp = (J * sigma_depth * sigma_depth * J).sum(dim=-1, keepdim=True)  # [h, w ,1]

        sigma_icp = torch.sqrt(cov_icp + 1e-8)
        return sigma_icp

    def compute_jtj(self, jac):
        # J in the dimension of (HW, C, 6)
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtj = torch.bmm(jacT, jac).sum(0)  # [6, 6]
        return jtj  # [6, 6]

    def compute_jtr(self, jac, res):
        # J in the dimension of (HW, C, 6)
        # res in the dimension of [HW, C]
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtr = torch.bmm(jacT, res.unsqueeze(-1)).sum(0)  # [6, 1]
        return jtr  # [6, 1]

    def GN_solver(self, JtJ, JtR, pose0, damping=1e-6):

        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ, damping)
        # Hessian = JtJ
        if self.direction == "forward":
            updated_pose = forward_update_pose(Hessian, JtR, pose0)
        else:
            updated_pose = inverse_update_pose(Hessian, JtR, pose0)

        return updated_pose

    def precompute_jacobian(self, depth0, f0, K, grad_interp=False, crd0=None):
        if not grad_interp:
            # inverse: no need for interpolation in gradients: (linearized at origin)
            f0_gradx, f0_grady = feature_gradient(f0, normalize_gradient=False)  # [H, W, C]
        else:
            # gradients of bilinear interpolation
            if crd0 is None:
                _, _, H, W = f0.shape
                crd0 = geometry.gen_coordinate_tensors(W, H).unsqueeze(dim=0).double()
            f0_gradx, f0_grady = geometry.grad_bilinear_interpolation(crd0, f0, replace_nan_as_eps=True)

        # grad_f0 = torch.stack((f0_gradx, f0_grady), dim=2)
        Jx_p, Jy_p = compute_jacobian_warping(depth0, K)  # [H, W, 6], [H, W, 6]
        J_F_p = f0_gradx.unsqueeze(-1) * Jx_p.unsqueeze(-2) + f0_grady.unsqueeze(-1) * Jy_p.unsqueeze(-2)
        return J_F_p  # [H, W, C, 6]

    def compute_inverse_residuals(self, pose10, vertex0, vertex1, x0, x1, K):
        R = pose10[:3, :3]
        t = pose10[:3, 3]
        H, W, C = vertex0.shape
        # warp (0) to (1)
        rot_vertex0_to1 = (R @ vertex0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)
        vertex0_to1 = rot_vertex0_to1 + t[None, None, :]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_, y_, z_ = vertex0_to1[..., 0], vertex0_to1[..., 1], vertex0_to1[..., 2]  # [h, w]
        u_ = (x_ / z_) * fx + cx  # [h, w]
        v_ = (y_ / z_) * fy + cy  # [h, w]
        r_x1 = warp_features(x1, u_, v_)  # [h, w, c]
        r_vertex1 = warp_features(vertex1, u_, v_, mode="nearest")
        res = r_x1 - x0  # [h, w, c]
        cv2.imwrite("img0.png", cv2.cvtColor(x0.cpu().numpy(), cv2.COLOR_RGB2BGR) * 255.)
        cv2.imwrite("img0to1.png", cv2.cvtColor(r_x1.cpu().numpy(), cv2.COLOR_RGB2BGR) * 255.)
        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)
        invalid_mask = ~inviews | (vertex0[..., -1] <= 0.) | (r_vertex1[..., -1] <= 0.)  # | (torch.norm(res, dim=-1) < 0.02)
        # handle invalidity
        res[invalid_mask, :] = 0.
        return res, invalid_mask


def warp_features(Feat, u, v, mode='bilinear'):
    """
    Warp the feature map (F) w.r.t. the grid (u, v). This is the non-batch version
    """
    assert len(Feat.shape) == 3
    H, W, C = Feat.shape
    u_norm = u / ((W - 1) / 2) - 1  # [h, w]
    v_norm = v / ((H - 1) / 2) - 1  # [h, w]
    uv_grid = torch.cat((u_norm.view(1, H, W, 1), v_norm.view(1, H, W, 1)), dim=-1)
    Feat_warped = F.grid_sample(Feat.unsqueeze(0).permute(0, 3, 1, 2), uv_grid, mode=mode, padding_mode='border', align_corners=True).squeeze()
    return Feat_warped.permute(1, 2, 0)


def compute_vertex(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    device = depth.device

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)  # [h, w]
    j = j.t().to(device)  # [h, w]

    vertex = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1).to(device) * depth[..., None]  # [h, w, 3]
    return vertex


def compute_normal(vertex_map):
    """ Calculate the normal map from a depth map
    :param the input depth image
    -----------
    :return the normal map
    """
    H, W, C = vertex_map.shape
    img_dx, img_dy = feature_gradient(vertex_map, normalize_gradient=False)  # [h, w, 3]

    normal = torch.cross(img_dx.view(-1, 3), img_dy.view(-1, 3))
    normal = normal.view(H, W, 3)  # [h, w, 3]

    mag = torch.norm(normal, p=2, dim=-1, keepdim=True)
    normal = normal / (mag + 1e-8)

    # filter out invalid pixels
    depth = vertex_map[:, :, -1]
    # 0.5 and 5.
    invalid_mask = (depth <= depth.min()) | (depth >= depth.max())
    zero_normal = torch.zeros_like(normal)
    normal = torch.where(invalid_mask[..., None], zero_normal, normal)

    return normal


def feature_gradient(img, normalize_gradient=True):
    """ Calculate the gradient on the feature space using Sobel operator
    :param the input image
    -----------
    :return the gradient of the image in x, y direction
    """
    H, W, C = img.shape
    # to filter the image equally in each channel
    wx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).type_as(img)
    wy = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).type_as(img)

    img_permuted = img.permute(2, 0, 1).view(-1, 1, H, W)  # [c, 1, h, w]
    # TODO: use this for computing normal fields
    img_pad = F.pad(img_permuted, (1, 1, 1, 1), mode='replicate')
    img_dx = F.conv2d(img_pad, wx, stride=1, padding=0).squeeze().permute(1, 2, 0)  # [h, w, c]
    img_dy = F.conv2d(img_pad, wy, stride=1, padding=0).squeeze().permute(1, 2, 0)  # [h, w, c]

    if normalize_gradient:
        mag = torch.sqrt((img_dx ** 2) + (img_dy ** 2) + 1e-8)
        img_dx = img_dx / mag
        img_dy = img_dy / mag

    return img_dx, img_dy  # [h, w, c]


def compute_jacobian_warping(depth, K, pose=None):
    """ Compute the Jacobian matrix of the warped (x,y) w.r.t. the inverse depth
    (linearized at origin)
    :param p_invdepth the input inverse depth
    :param the intrinsic calibration
    :param the pixel x map
    :param the pixel y map
     ------------
    :return the warping jacobian in x, y direction
    """
    assert len(depth.shape) == 2
    H, W = depth.size()

    # if pose is not None:
    #     x_y_invz = torch.cat((px, py, p_invdepth), dim=1)
    #     R, t = pose
    #     warped = torch.bmm(R, x_y_invz.view(B, 3, H * W)) + \
    #              t.view(B, 3, 1).expand(B, 3, H * W)
    #     px, py, p_invdepth = warped.split(1, dim=1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    device = depth.device

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)  # [h, w]
    j = j.t().to(device)  # [h, w]
    x = (i - cx) / fx * depth
    y = (j - cy) / fy * depth
    invD = 1. / depth
    invD2 = invD ** 2

    xy = x * y
    O = torch.zeros((H, W)).type_as(depth)
    # This is cascaded Jacobian functions of the warping function
    # Refer to the supplementary materials for math documentation
    dx_dp = torch.stack((-invD2 * xy, 1 + x ** 2 * invD2, -y * invD, invD, O, -invD2 * x), dim=-1)
    dy_dp = torch.stack((-1 - invD2 * y ** 2, xy * invD2, x * invD, O, invD, -invD2 * y), dim=-1)

    return dx_dp * fx, dy_dp * fy


def batch_skew(w):
    """ Generate a batch of skew-symmetric matrices.

        function tested in 'test_geometry.py'

    :input
    :param skew symmetric matrix entry Bx3
    ---------
    :return
    :param the skew-symmetric matrix Bx3x3
    """
    B, D = w.shape
    assert(D == 3)
    o = torch.zeros(B).type_as(w)
    w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
    return torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), 1).view(B, 3, 3)


def lev_mar_H(JtWJ, damping):
    # Add a small diagonal damping. Without it, the training becomes quite unstable
    # Do not see a clear difference by removing the damping in inference though
    diag_mask = torch.eye(6).to(JtWJ)
    diagJtJ = diag_mask * JtWJ
    traceJtJ = torch.sum(diagJtJ)
    epsilon = (traceJtJ * damping) * diag_mask
    Hessian = JtWJ + epsilon
    return Hessian


def forward_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update
    in the forward compositional form
    ksi_k o (delta_ksi)
    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """
    xi = least_square_solve(H, Rhs).squeeze()

    # forward compotional for SE3: delta_ksi
    d_R = exp_so3(xi[:3])
    d_t = xi[3:]
    R = pose[:3, :3]
    t = pose[:3, 3]
    R1 = d_R @ R
    t1 = d_R @ t + d_t
    pose[:3, :3] = R1
    pose[:3, 3] = t1

    # my implementation using full SE(3) exponential
    # pose = exp_se3(xi) @ pose
    return pose


def inverse_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update
    in the forward compositional form
    ksi_k o (delta_ksi)
    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """
    xi = least_square_solve(H, Rhs).squeeze()

    # forward compotional for SE3: delta_ksi
    d_R = exp_so3(-xi[:3])
    d_t = -d_R @ xi[3:]
    R = pose[:3, :3]
    t = pose[:3, 3]
    R1 = d_R @ R
    t1 = d_R @ t + d_t
    pose[:3, :3] = R1
    pose[:3, 3] = t1

    # my implementation using full SE(3) exponential
    # pose = exp_se3(xi) @ pose
    return pose


def exp_so3(w):
    w_hat = torch.tensor([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]]).to(w)
    w_hat_second = torch.mm(w_hat, w_hat).to(w)

    theta = torch.norm(w)
    theta_2 = theta ** 2
    theta_3 = theta ** 3
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    eye_3 = torch.eye(3).to(w)

    eps = 1e-10

    if theta <= eps:
        e_w = eye_3
    else:
        e_w = eye_3 + w_hat * sin_theta / theta + w_hat_second * (1. - cos_theta) / theta_2

    return e_w


def exp_se3(xi):
    """
    :param x: Cartesian vector of Lie Algebra se(3)
    :return: exponential map of x
    """
    w = xi[:3].squeeze()  # rotation
    v = xi[3:6].squeeze()  # translation
    w_hat = torch.tensor([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]]).to(xi)
    w_hat_second = torch.mm(w_hat, w_hat).to(xi)

    theta = torch.norm(w)
    theta_2 = theta ** 2
    theta_3 = theta ** 3
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    eye_3 = torch.eye(3).to(xi)

    eps = 1e-8

    if theta <= eps:
        e_w = eye_3
        j = eye_3
    else:
        e_w = eye_3 + w_hat * sin_theta / theta + w_hat_second * (1. - cos_theta) / theta_2
        k1 = (1 - cos_theta) / theta_2
        k2 = (theta - sin_theta) / theta_3
        j = eye_3 + k1 * w_hat + k2 * w_hat_second

    T = torch.eye(4).to(xi)
    T[:3, :3] = e_w
    # T[:3, 3] = torch.mv(j, v)
    T[:3, 3] = v

    return T


def invH(H):
    """ Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    # GPU is much slower for matrix inverse when the size is small (compare to CPU)
    # works (50x faster) than inversing the dense matrix in GPU
    if H.is_cuda:
        # invH = bpinv((H).cpu()).cuda()
        # invH = torch.inverse(H)
        invH = torch.inverse(H.cpu()).cuda()
    else:
        invH = torch.inverse(H)
    return invH


def least_square_solve(H, Rhs):
    """
    x =  - H ^{-1} * Rhs
    importantly: use pytorch inverse to have a differential inverse operation
    :param H: Hessian
    :type H: [6, 6]
    :param  Rhs: Right-hand side vector
    :type Rhs: [6, 1]
    :return: solved ksi
    :rtype:  [6, 1]
    """
    inv_H = invH(H)  # [B, 6, 6] square matrix
    xi = inv_H @ Rhs
    # because the jacobian is also including the minus signal, it should be (J^T * J) J^T * r
    # xi = - xi
    return xi


def huber_norm_weights(x, b=0.02):
    """
    :param x: norm of residuals, torch.Tensor (N, 1)
    :param b: threshold
    :return: weight vector torch.Tensor (N, 1)
    """
    # x is residual norm
    res_norm = torch.zeros_like(x)
    res_norm[x <= b] = x[x <= b] ** 2
    res_norm[x > b] = 2 * b * x[x > b] - b ** 2
    x[x < 1e-8] = 1.
    return torch.sqrt(res_norm) / x


def get_robust_res(res, b):
    """
    :param res: residual vectors [N, C]
    :param b: threshold
    :return: residuals after applying huber norm
    """
    assert len(res.shape) == 2
    res_norm = torch.norm(res, dim=-1, keepdim=True)
    # print(res.shape[0])
    w = huber_norm_weights(res_norm, b=b)
    # print(w.shape[0])
    robust_res = w * res

    return robust_res, w