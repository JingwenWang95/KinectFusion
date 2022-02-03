import os
import argparse
import numpy as np
import torch
import trimesh
import open3d as o3d
from matplotlib import pyplot as plt
from fusion import TSDFVolumeTorch
from dataset.TUM_RGBD import TUMDataset, TUMDatasetOnline
from Tracker import ICPTracker
from utils import load_config, get_volume_setting, get_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default="configs/fr1_room.yaml", help='Path to config file.')
    args = load_config(parser.parse_args())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUMDatasetOnline(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25)
    H, W, K = dataset.H, dataset.W, dataset.K

    vol_dims, vol_origin, voxel_size = get_volume_setting(args)
    tsdf_volume = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True)
    icp_tracker = ICPTracker(args, device)

    t, poses, poses_gt = list(), list(), list()
    curr_pose, depth1, color1 = None, None, None
    for i in range(0, len(dataset), 1):
        t0 = get_time()
        sample = dataset[i]
        color0, depth0, pose_gt, K = sample  # use live image as template image (0)
        # depth0[depth0 <= 0.5] = 0.

        if i == 0:  # initialize
            curr_pose = pose_gt
        else:  # tracking
            # 1. render depth image (1) from tsdf volume
            depth1, color1, vertex01, normal1, mask1 = tsdf_volume.render_model(curr_pose, K, H, W, near=args.near, far=args.far, n_samples=args.n_steps)
            T10 = icp_tracker(depth0, depth1, K)  # transform from 0 to 1
            curr_pose = curr_pose @ T10

        # fusion
        tsdf_volume.integrate(depth0,
                              K,
                              curr_pose,
                              obs_weight=1.,
                              color_img=color0
                              )
        t1 = get_time()
        t += [t1 - t0]
        print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
        poses += [curr_pose.cpu().numpy()]
        poses_gt += [pose_gt.cpu().numpy()]

    avg_time = np.array(t).mean()
    print("average processing time: {:f}s per frame, i.e. {:f} fps".format(avg_time, 1. / avg_time))

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

