import os
import numpy as np
import torch
import trimesh
import cv2
import open3d as o3d
from my_fusion import TSDFVolumeTorch, MeshRenderer
from data_slam.TUM_RGBD import TUMDataset


def get_intri_from_calib(calib):
    intri = np.eye(3)
    intri[0, 0] = calib[0]
    intri[1, 1] = calib[1]
    intri[0, 2] = calib[2]
    intri[1, 2] = calib[3]
    return intri


if __name__ == "__main__":
    data_root = "/home/jingwen/data/tum_rgbd"
    exp_dir = "../../logs/kf_vo/retrained"
    seq_prefix = 'rgbd_dataset_'
    seq = 'freiburg1_desk'
    sequence_dir = seq_prefix + seq
    N = 500
    near = 0.2
    far = 5.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUMDataset(os.path.join(data_root, sequence_dir), device, near=near, far=far, end=N, img_scale=0.25)
    H, W = dataset.H, dataset.W
    vol_size = 256
    voxel_size = 2.0 / vol_size
    # fr1_desk
    vol_bnds = np.array([[-3., 2.],
                         [-1., 2.],
                         [-0.2, 2.5]])

    # fr1_360
    # vol_bnds = np.array([[-3.5, 3.5],
    #                      [-3.5, 3.5],
    #                      [-0.2, 3.5]])

    tsdf_volume = TSDFVolumeTorch([vol_size, vol_size, vol_size // 2], [-1, -1, -1], voxel_size, margin=3, vol_bnds=vol_bnds, fuse_color=True, use_gpu=True)
    poses = []
    depths = []
    for i in range(0, N, 1):
        sample = dataset[i]
        color, depth, pose, intrinsic = sample
        # depth[depth <= 0.5] = -1.
        color *= 255.
        extrinsic = pose
        tsdf_volume.integrate(depth,
                              intrinsic,
                              extrinsic,
                              obs_weight=1.,
                              color_img=color,
                              feat_img=None
                              )
        print("processed frame: {:d}".format(i))
        poses += [pose]
        depths += [depth]

    verts, faces, norms, colors = tsdf_volume.get_mesh()
    partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
    partial_tsdf.export("mesh.ply")

    # render the fused model
    rgbdir = "rend_imgs/{:s}/rgb".format(seq)
    if not os.path.exists(rgbdir):
        os.makedirs(rgbdir)
    depthdir = "rend_imgs/{:s}/depth".format(seq)
    if not os.path.exists(depthdir):
        os.makedirs(depthdir)
    errdir = "rend_imgs/{:s}/error".format(seq)
    if not os.path.exists(errdir):
        os.makedirs(errdir)

    mesh_renderer = MeshRenderer(H, W, intrinsic.cpu().numpy())
    mesh = o3d.io.read_triangle_mesh("mesh.ply")
    mesh_renderer.render(mesh, poses[0].cpu().numpy())

    depth_errs = []
    depth_diffs = []
    for i in range(0, N, 1):
        c2w = poses[i]
        intri = intrinsic
        rend_depth, rend_color, _, _, _, hit_surface_mask = tsdf_volume.render_model(c2w, intri, color.shape[0], color.shape[1], near=near, far=far, n_samples=192)
        # rend_depth, _ = mesh_renderer.render(mesh, poses[i].cpu().numpy())
        rend_color = cv2.cvtColor(rend_color.cpu().numpy(), cv2.COLOR_RGB2BGR)
        rend_depth = rend_depth.cpu().numpy()
        # compute depth error
        depth_gt = depths[i].cpu().numpy()
        # depth_err = np.mean(np.abs(depth_gt[depth_gt > 0.] - rend_depth[depth_gt > 0.]))
        # print(depth_err)
        # depth_errs += [depth_err]
        hit_surface_mask = hit_surface_mask.cpu().numpy()
        surf_depth_gt = depth_gt[hit_surface_mask]
        surf_depth_rend = rend_depth[hit_surface_mask]
        depth_err = np.mean(np.abs(surf_depth_gt[surf_depth_gt > 0.] - surf_depth_rend[surf_depth_gt > 0.]))
        depth_diff = np.mean(surf_depth_gt[surf_depth_gt > 0.] - surf_depth_rend[surf_depth_gt > 0.])
        print("abs: {:f}, diff: {:f}".format(depth_err, depth_diff))
        depth_errs += [depth_err]
        depth_diffs += [depth_diff]
        err_map = np.zeros_like(depth_gt)
        surface_depth_err = np.abs(surf_depth_gt - surf_depth_rend)
        surface_depth_err[surf_depth_gt < 0.] = 0.
        err_map[hit_surface_mask] = surface_depth_err
        err_map /= err_map.max()
        cv2.imwrite(os.path.join(rgbdir, "{:04d}.png".format(i)), rend_color)
        cv2.imwrite(os.path.join(depthdir, "{:04d}.png".format(i)), (rend_depth * 5000.).astype(np.uint16))
        cv2.imwrite(os.path.join(errdir, "{:04d}.png".format(i)), (err_map * 255).astype(np.uint8))

    depth_errs = np.array(depth_errs)
    depth_diffs = np.array(depth_diffs)
    print("Mean depth error: {:f} meters".format(depth_errs.mean()))
    print("Mean depth bias: {:f} meters".format(depth_diffs.mean()))
    # x = torch.Tensor([[1., 1., 1.], [0., 0., 0.]])
    # id = tsdf_volume.get_voxel_idx(x)
    # print(tsdf_volume.world_c[id])
    # print(tsdf_volume.vox_coords[id])
