import os
import numpy as np
import torch
import trimesh
import open3d as o3d
from matplotlib import pyplot as plt
from fusion import TSDFVolumeTorch
from dataset.TUM_RGBD import TUMDataset
from Tracker import ICPTracker


if __name__ == "__main__":
    data_root = "/home/jingwen/data/tum_rgbd"
    exp_dir = "../../logs/kf_vo/retrained"
    seq_prefix = 'rgbd_dataset_'
    seq = 'freiburg1_desk'
    # seq = "freiburg1_360"
    # seq = 'freiburg3_long_office_household'
    sequence_dir = seq_prefix + seq
    N = -1

    near = 0.1
    far = 5.
    voxel_size = 0.02
    xmin, xmax = -2.5, 2.
    ymin, ymax = -0.7, 2.7
    zmin, zmax = -0.2, 2.  # fr1_desk
    vol_bnds = np.array([[xmin, xmax],
                         [ymin, ymax],
                         [zmin, zmax]])
    vol_dims = list((vol_bnds[:, 1] - vol_bnds[:, 0]) // voxel_size)
    vol_origin = list(vol_bnds[:, 0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUMDataset(os.path.join(data_root, sequence_dir), device, near=near, far=far, img_scale=0.25, start=0, end=N)
    H, W, K = dataset.H, dataset.W, dataset.K
    N = len(dataset)

    tsdf_volume = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, vol_bnds=None, fuse_color=True)
    icp_tracker = ICPTracker(3, device)
    poses = []
    poses_gt = []
    curr_pose = None  # c2w Twc
    depth1 = None
    color1 = None
    for i in range(0, N, 1):
        sample = dataset[i]
        color0, depth0, pose_gt, K = sample  # use live image as template image (0)
        # depth0[depth0 <= 0.5] = 0.
        color0 *= 255.

        if i == 0:  # initialize
            curr_pose = pose_gt
        else:  # tracking
            # 1. render depth image (1) from tsdf volume
            depth1, color1, vertex01, normal1, mask1 = tsdf_volume.render_model(curr_pose, K, H, W, n_samples=192)
            T10 = icp_tracker(depth0, depth1, K)  # transform from 0 to 1
            curr_pose = curr_pose @ T10

        # fusion
        tsdf_volume.integrate(depth0,
                              K,
                              curr_pose,
                              obs_weight=1.,
                              color_img=color0
                              )
        # depth1 = depth0
        # color1 = color0
        print("processed frame: {:d}".format(i))
        poses += [curr_pose.cpu().numpy()]
        poses_gt += [pose_gt.cpu().numpy()]

    verts, faces, norms, colors = tsdf_volume.get_mesh()
    partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
    partial_tsdf.export("mesh.ply")
    np.savez("traj.npz", poses=np.stack(poses, 0))
    np.savez("traj_gt.npz", poses=np.stack(poses_gt, 0))

    gt_poses = np.load(os.path.join("traj_gt.npz"))["poses"]
    poses = np.load(os.path.join("traj.npz"))["poses"]
    traj_gt = np.array(gt_poses)[:, :3, 3]
    traj = np.array(poses)[:, :3, 3]
    rmse = np.sqrt(np.mean(np.linalg.norm(traj_gt - traj, axis=-1) ** 2))
    print("RMSE: {:f}".format(rmse))
    plt.plot(traj[:, 0], traj[:, 1])
    plt.plot(traj_gt[:, 0], traj_gt[:, 1])
    plt.legend(['Estimated', 'GT'])
    plt.show()

