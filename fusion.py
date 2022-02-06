import os
import numpy as np
from skimage import measure
import torch
import cv2
import open3d as o3d
import imageio


def integrate(
        depth_im,
        cam_intr,
        cam_pose,
        obs_weight,
        world_c,  # world coordinates grid [nx*ny*nz, 4]
        vox_coords,  # voxel coordinates grid [nx*ny*nz, 3]
        weight_vol,  # weight volume [nx, ny, nz]
        tsdf_vol,  # tsdf volume [nx, ny, nz]
        sdf_trunc,
        im_h,
        im_w,
        color_vol=None,
        color_im=None,
):

    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()  # [nx*ny*nz, 4]
    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    # project all the voxels back to image plane
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()  # [nx*ny*nz]
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()  # [nx*ny*nz]

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)  # [n_valid]
    valid_vox_x = vox_coords[valid_pix, 0]
    valid_vox_y = vox_coords[valid_pix, 1]
    valid_vox_z = vox_coords[valid_pix, 2]
    depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]  # [n_valid]

    # Integrate tsdf
    depth_diff = depth_val - pix_z[valid_pix]
    dist = torch.clamp(depth_diff / sdf_trunc, max=1)
    valid_pts = (depth_val > 0.) & (depth_diff >= -sdf_trunc)  # all points 1. inside frustum 2. with valid depth 3. outside -truncate_dist
    valid_vox_x = valid_vox_x[valid_pts]
    valid_vox_y = valid_vox_y[valid_pts]
    valid_vox_z = valid_vox_z[valid_pts]
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    if color_vol is not None and color_im is not None:
        old_color = color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        new_color = color_im[pix_y[valid_pix], pix_x[valid_pix]]
        new_color = new_color[valid_pts]
        color_vol[valid_vox_x, valid_vox_y, valid_vox_z, :] = (w_old[:, None] * old_color + obs_weight * new_color) / w_new[:, None]

    return weight_vol, tsdf_vol, color_vol


class TSDFVolumeTorch:
    """
    Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, voxel_dim, origin, voxel_size, device, margin=3, fuse_color=False):
        """
        Args:
            voxel_dim (ndarray): [3,] stores volume dimensions: Nx, Ny, Nz
            origin (ndarray): [3,] world coordinate of voxel [0, 0, 0]
            voxel_size (float): The volume discretization in meters.
        """

        self.device = device
        # Define voxel volume parameters
        self.voxel_size = float(voxel_size)
        self.sdf_trunc = margin * self.voxel_size
        self.integrate_func = integrate
        self.fuse_color = fuse_color

        # Adjust volume bounds
        if isinstance(voxel_dim, list):
            voxel_dim = torch.Tensor(voxel_dim).to(self.device)
        elif isinstance(voxel_dim, np.ndarray):
            voxel_dim = torch.from_numpy(voxel_dim).to(self.device)
        if isinstance(origin, list):
            origin = torch.Tensor(origin).to(self.device)
        elif isinstance(origin, np.ndarray):
            origin = torch.from_numpy(origin).to(self.device)

        self.vol_dim = voxel_dim.long()
        self.vol_origin = origin
        self.num_voxels = torch.prod(self.vol_dim).item()

        # Get voxel grid coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self.vol_dim[0]),
            torch.arange(0, self.vol_dim[1]),
            torch.arange(0, self.vol_dim[2]),
        )
        self.vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)

        # Convert voxel coordinates to world coordinates
        self.world_c = self.vol_origin + (self.voxel_size * self.vox_coords)
        self.world_c = torch.cat([
            self.world_c, torch.ones(len(self.world_c), 1, device=self.device)], dim=1).float()
        self.reset()

    def reset(self):
        """Set volumes
        """
        self.tsdf_vol = torch.ones(*self.vol_dim).to(self.device)
        self.weight_vol = torch.zeros(*self.vol_dim).to(self.device)
        if self.fuse_color:
            # [nx, ny, nz, 3]
            self.color_vol = torch.zeros(*self.vol_dim, 3).to(self.device)
        else:
            self.color_vol = None

    def data_transfer(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.float().to(self.device)

    @torch.no_grad()
    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight, color_img=None):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
        depth_im (torch.Tensor): A depth image of shape (H, W).
        cam_intr (torch.Tensor): The camera intrinsics matrix of shape (3, 3).
        cam_pose (torch.Tensor): The camera pose (i.e. extrinsics) of shape (4, 4). T_wc
        obs_weight (float): The weight to assign to the current observation.
        """

        cam_pose = self.data_transfer(cam_pose)
        cam_intr = self.data_transfer(cam_intr)
        depth_im = self.data_transfer(depth_im)
        if color_img is not None:
            color_img = self.data_transfer(color_img)
        else:
            color_img = None
        im_h, im_w = depth_im.shape
        # fuse
        weight_vol, tsdf_vol, color_vol = self.integrate_func(
            depth_im,
            cam_intr,
            cam_pose,
            obs_weight,
            self.world_c,
            self.vox_coords,
            self.weight_vol,
            self.tsdf_vol,
            self.sdf_trunc,
            im_h, im_w,
            self.color_vol,
            color_img,
        )
        self.weight_vol = weight_vol
        self.tsdf_vol = tsdf_vol
        self.color_vol = color_vol

    def get_volume(self):
        return self.tsdf_vol, self.weight_vol, self.color_vol

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, weight_vol, color_vol = self.get_volume()
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol.cpu().numpy(), level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.voxel_size + self.vol_origin.cpu().numpy()  # voxel grid coordinates to world coordinates

        if self.fuse_color:
            rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].cpu().numpy()
            return verts, faces, norms, rgb_vals.astype(np.uint8)
        else:
            return verts, faces, norms

    def to_o3d_mesh(self):
        """Convert to o3d mesh object for visualization
        """
        verts, faces, norms, colors = self.get_mesh()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(float))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.)
        return mesh

    def get_normals(self):
        """Compute normal volume
        """
        nx, ny, nz = self.vol_dim
        device = self.device
        # dx = torch.cat([torch.zeros(1, ny, nz).to(device), (self.tsdf_vol[2:, :, :] - self.tsdf_vol[:-2, :, :]) / (2 * self.voxel_size), torch.zeros(1, ny, nz).to(device)], dim=0)
        # dy = torch.cat([torch.zeros(nx, 1, nz).to(device), (self.tsdf_vol[:, 2:, :] - self.tsdf_vol[:, :-2, :]) / (2 * self.voxel_size), torch.zeros(nx, 1, nz).to(device)], dim=1)
        # dz = torch.cat([torch.zeros(nx, ny, 1).to(device), (self.tsdf_vol[:, :, 2:] - self.tsdf_vol[:, :, :-2]) / (2 * self.voxel_size), torch.zeros(nx, ny, 1).to(device)], dim=2)
        # norms = torch.stack([dx, dy, dz], -1)
        dx = torch.cat([(self.tsdf_vol[1:, :, :] - self.tsdf_vol[:-1, :, :]) / self.voxel_size, torch.zeros(1, ny, nz).to(device)], dim=0)
        dy = torch.cat([(self.tsdf_vol[:, 1:, :] - self.tsdf_vol[:, :-1, :]) / self.voxel_size, torch.zeros(nx, 1, nz).to(device)], dim=1)
        dz = torch.cat([(self.tsdf_vol[:, :, 1:] - self.tsdf_vol[:, :, :-1]) / self.voxel_size, torch.zeros(nx, ny, 1).to(device)], dim=2)
        norms = torch.stack([dx, dy, dz], -1)
        n = torch.norm(norms, dim=-1)
        # remove large values
        outliers_mask = n > 1. / (2 * self.voxel_size)
        norms[outliers_mask] = 0.
        # normalize
        eps = 1e-7
        non_zero_grad = n > eps
        norms[non_zero_grad, :] = norms[non_zero_grad, :] / n[non_zero_grad][:, None]
        return norms  # [nx, ny, nz, 3]

    def get_nn(self, field_vol, coords_w):
        """Get nearest-neigbor values from a given volume
        """
        field_dim = field_vol.shape
        assert len(field_dim) == 3 or len(field_dim) == 4
        vox_coord_float = (coords_w - self.vol_origin[None, :]) / self.voxel_size
        vox_coord = torch.floor(vox_coord_float)
        vox_offset = vox_coord_float - vox_coord  # [N, 3]
        vox_coord[vox_offset >= 0.5] += 1.
        vox_coord[:, 0] = torch.clamp(vox_coord[:, 0], 0., self.vol_dim[0] - 1)
        vox_coord[:, 1] = torch.clamp(vox_coord[:, 1], 0., self.vol_dim[1] - 1)
        vox_coord[:, 2] = torch.clamp(vox_coord[:, 2], 0., self.vol_dim[2] - 1)
        vox_coord = vox_coord.long()
        vx, vy, vz = vox_coord[:, 0], vox_coord[:, 1], vox_coord[:, 2]
        v_nn = field_vol[vx, vy, vz]
        return v_nn

    def tril_interp(self, field_vol, coords_w):
        """Get tri-linear interpolated value from a given volume
        """
        field_dim = field_vol.shape
        assert len(field_dim) == 3 or len(field_dim) == 4
        n_pts = coords_w.shape[0]
        vox_coord = torch.floor((coords_w - self.vol_origin[None, :]) / self.voxel_size).long()  # [N, 3]

        # for border points, don't do interpolation
        non_border_mask = (vox_coord[:, 0] < self.vol_dim[0] - 1) & (vox_coord[:, 1] < self.vol_dim[1] - 1) & \
                          (vox_coord[:, 2] < self.vol_dim[2] - 1)
        v_interp = torch.zeros(n_pts) if len(field_dim) == 3 else torch.zeros(n_pts, field_vol.shape[-1])
        v_interp = v_interp.to(self.device)
        vx_, vy_, vz_ = vox_coord[~non_border_mask, 0], vox_coord[~non_border_mask, 1], vox_coord[~non_border_mask, 2]
        v_interp[~non_border_mask] = field_vol[vx_, vy_, vz_]

        # get interpolated values for normal points
        vx, vy, vz = vox_coord[non_border_mask, 0], vox_coord[non_border_mask, 1], vox_coord[non_border_mask, 2]  # [N]
        vox_idx = vz + vy * self.vol_dim[-1] + vx * self.vol_dim[-1] * self.vol_dim[-2]
        vertices_coord = self.world_c[vox_idx][:, :3]  # [N, 3]
        r = (coords_w[non_border_mask] - vertices_coord) / self.voxel_size
        rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
        if len(field_dim) == 4:
            rx = rx.unsqueeze(1)
            ry = ry.unsqueeze(1)
            rz = rz.unsqueeze(1)
        # get values at eight corners
        v000 = field_vol[vx, vy, vz]
        v001 = field_vol[vx, vy, vz+1]
        v010 = field_vol[vx, vy+1, vz]
        v011 = field_vol[vx, vy+1, vz+1]
        v100 = field_vol[vx+1, vy, vz]
        v101 = field_vol[vx+1, vy, vz+1]
        v110 = field_vol[vx+1, vy+1, vz]
        v111 = field_vol[vx+1, vy+1, vz+1]
        v_interp[non_border_mask] = v000 * (1 - rx) * (1 - ry) * (1 - rz) \
                                   + v001 * (1 - rx) * (1 - ry) * rz \
                                   + v010 * (1 - rx) * ry * (1 - rz) \
                                   + v011 * (1 - rx) * ry * rz \
                                   + v100 * rx * (1 - ry) * (1 - rz) \
                                   + v101 * rx * (1 - ry) * rz \
                                   + v110 * rx * ry * (1 - rz) \
                                   + v111 * rx * ry * rz

        return v_interp

    def get_pts_inside(self, pts, margin=0):
        vox_coord = torch.floor((pts - self.vol_origin[None, :]) / self.voxel_size).long()  # [N, 3]
        valid_pts_mask = (vox_coord[..., 0] >= margin) & (vox_coord[..., 0] < self.vol_dim[0] - margin) \
                         & (vox_coord[..., 1] >= margin) & (vox_coord[..., 1] < self.vol_dim[1] - margin) \
                         & (vox_coord[..., 2] >= margin) & (vox_coord[..., 2] < self.vol_dim[2] - margin)
        return valid_pts_mask

    # use simple root finding
    @torch.no_grad()
    def render_model(self, c2w, intri, imh, imw, near=0.5, far=5., n_samples=192):
        """
        Perform ray-casting for frame-to-model tracking
        :param c2w: camera pose, [4, 4]
        :param intri: camera intrinsics, [3, 3]
        :param imh: image height
        :param imw: image width
        :param near: near bound for ray-casting
        :param far: far bound for ray-casting
        :param n_samples: number of samples along the ray
        :return: rendered depth, color, vertex, normal and valid mask, [H, W, C]
        """
        rays_o, rays_d = self.get_rays(c2w, intri, imh, imw)  # [h, w, 3]
        z_vals = torch.linspace(near, far, n_samples).to(rays_o)  # [n_samples]
        ray_pts_w = (rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_vals[None, None, :, None]).to(self.device)  # [h, w, n_samples, 3]

        # need to query the tsdf and feature grid
        tsdf_vals = torch.ones(imh, imw, n_samples).to(self.device)
        # filter points that are outside the volume
        valid_ray_pts_mask = self.get_pts_inside(ray_pts_w)
        valid_ray_pts = ray_pts_w[valid_ray_pts_mask]  # [n_valid, 3]
        tsdf_vals[valid_ray_pts_mask] = self.tril_interp(self.tsdf_vol, valid_ray_pts)

        # surface prediction by finding zero crossings
        sign_matrix = torch.cat([torch.sign(tsdf_vals[..., :-1] * tsdf_vals[..., 1:]),
                                 torch.ones(imh, imw, 1).to(self.device)], dim=-1)  # [h, w, n_samples]
        cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float().to(self.device)[None, None, :]  # [h, w, n_samples]
        # Get first sign change and mask for values where
        # a.) a sign changed occurred and
        # b.) not a neg to pos sign change occurred
        # c.) ignore border points
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        hs, ws = torch.meshgrid(torch.arange(imh), torch.arange(imw))
        mask_pos_to_neg = tsdf_vals[hs, ws, indices] > 0
        inside_vol = self.get_pts_inside(ray_pts_w[hs, ws, indices])
        hit_surface_mask = mask_sign_change & mask_pos_to_neg & inside_vol
        hit_pts = ray_pts_w[hs, ws, indices][hit_surface_mask]  # [n_surf_pts, 3]

        # compute normals
        norms = self.get_normals()
        surf_tsdf = self.tril_interp(self.tsdf_vol, hit_pts)  # [n_surf_pts]
        # surf_norms = self.tril_interp(norms, hit_pts)  # [n_surf_pts, 3]
        surf_norms = self.get_nn(norms, hit_pts)
        updated_hit_pts = hit_pts - surf_tsdf[:, None] * self.sdf_trunc * surf_norms
        valid_mask = self.get_pts_inside(updated_hit_pts)
        hit_pts[valid_mask, :] = updated_hit_pts[valid_mask, :]

        # get depth values
        w2c = torch.inverse(c2w).to(self.device)
        hit_pts_c = (w2c[:3, :3] @ hit_pts.transpose(1, 0)).transpose(1, 0) + w2c[:3, 3][None, :]
        hit_pts_z = hit_pts_c[:, -1]
        depth_rend = torch.zeros(imh, imw).to(self.device)
        # depth_rend[hit_surface_mask] = z_vals[indices[hit_surface_mask]]
        depth_rend[hit_surface_mask] = hit_pts_z

        # vertex map
        vertex_rend = torch.zeros(imh, imw, 3).to(self.device)
        vertex_rend[hit_surface_mask] = hit_pts_c
        # normal map
        surf_norms_c = (w2c[:3, :3] @ surf_norms.transpose(1, 0)).transpose(1, 0)  # [h, w, 3]
        normal_rend = torch.zeros(imh, imw, 3).to(self.device)
        normal_rend[hit_surface_mask] = surf_norms_c

        if self.color_vol is not None:
            # hit_colors = self.color_vol[cx, cy, cz, :]
            hit_colors = self.tril_interp(self.color_vol, hit_pts)
            # set color
            color_rend = torch.zeros(imh, imw, 3).to(self.device)
            color_rend[hit_surface_mask] = hit_colors
        else:
            color_rend = None

        return depth_rend, color_rend, vertex_rend, normal_rend, hit_surface_mask

    def render_pyramid(self, c2w, intri, imh, imw, n_pyr=4, near=0.5, far=5., n_samples=192):
        K = intri.clone()
        dep_pyr, rgb_pyr, vtx_pyr, nrm_pyr, mask_pyr = [], [], [], [], []
        for l in range(n_pyr):
            dep, rgb, feat, vtx, nrm, mask = self.render_model(c2w, K, imh, imw, near=near, far=far, n_samples=n_samples)
            dep_pyr += [dep]
            rgb_pyr += [rgb]
            vtx_pyr += [vtx]
            nrm_pyr += [nrm]
            mask_pyr += [mask]
            imh = imh // 2
            imw = imw // 2
            K /= 2
        return dep_pyr, rgb_pyr, vtx_pyr, nrm_pyr, mask_pyr

    # get voxel index given world coordinate
    # used for testing
    def get_voxel_idx(self, x):
        """
        :param x: [N, 3] query points
        :return: [N] voxel indices
        """
        assert len(x.shape) == 2, print("only accept flattened input!!!")
        x.to(self.device)
        vox_coord = torch.floor((x - self.vol_origin[None, :]) / self.voxel_size)  # [N, 3]
        vx, vy, vz = vox_coord[:, 0], vox_coord[:, 1], vox_coord[:, 2]
        # very important! get voxel index from voxel coordinate
        vox_idx = vz + vy * self.vol_dim[-1] + vx * self.vol_dim[-1] * self.vol_dim[-2]
        return vox_idx.long()

    def get_rays(self, c2w, intrinsics, H, W):
        device = self.device
        c2w = c2w.to(device)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t().to(device).reshape(H * W)  # [hw]
        j = j.t().to(device).reshape(H * W)  # [hw]

        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1).to(device)  # [hw, 3]
        # permute for bmm
        dirs = dirs.transpose(1, 0)  # [3, hw]
        rays_d = (c2w[:3, :3] @ dirs).transpose(1, 0)  # [hw, 3]
        rays_o = c2w[:3, 3].expand(rays_d.shape)

        return rays_o.reshape(H, W, 3), rays_d.reshape(H, W, 3)
