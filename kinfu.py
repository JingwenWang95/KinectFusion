import os
import numpy as np
import torch
import trimesh
import open3d as o3d
from matplotlib import pyplot as plt
from my_fusion import TSDFVolumeTorch, MeshRenderer
from data_slam.TUM_RGBD import TUMDataset
from Tracker import ICPTracker, ColorICPTracker


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUMDataset(os.path.join(data_root, sequence_dir), device, near=near, far=far, img_scale=0.25, start=0, end=N)
    H, W, K = dataset.H, dataset.W, dataset.K
    N = len(dataset)

    # vol_size = 128
    # voxel_size = 2.0 / vol_size
    # vol_bnds = np.array([[-3., 2.],
    #                      [-1., 2.],
    #                      [-0.2, 2.5]])  # fr1_desk

    # vol_size = 180
    # voxel_size = 2. / vol_size
    # vol_bnds = np.array([[-3, 3.],
    #                      [-3., 4.5],
    #                      [-0.2, 3.5]])  # fr1_360

    vol_size = 128
    voxel_size = 2.0 / vol_size
    vol_bnds = np.array([[-2.5, 1.],
                         [-1.5, 1.5],
                         [-0.2, 2.0]])  # fr3_long_offcie

    tsdf_volume = TSDFVolumeTorch([vol_size, vol_size, vol_size], [-1, -1, -1], voxel_size, margin=3, vol_bnds=vol_bnds, fuse_color=True, use_gpu=True)
    icp_tracker = ICPTracker(3, device, direction="backward")
    coloricp_tracker = ColorICPTracker(3, device)
    mesh_renderer = MeshRenderer(H, W, K.cpu().numpy())
    poses = []
    poses_gt = []
    curr_pose = None  # c2w Twc
    depth1 = None
    color1 = None
    for i in range(0, N, 1):
        sample = dataset[i]
        color0, depth0, pose_gt, K = sample  # make live image as template image
        # depth0[depth0 <= 0.5] = 0.
        color0 *= 255.

        if i == 0:  # initialize
            curr_pose = pose_gt
        else:  # tracking
            # 1. render tsdf volume
            depth1, color1, _, vertex01, normal1, mask1 = tsdf_volume.render_model(curr_pose, K, H, W, n_samples=192)
            # T10 = coloricp_tracker(depth0, depth1, color0 / 255., color1 / 255., K)
            T10 = icp_tracker(depth0, depth1, K)
            curr_pose = curr_pose @ T10

            # 2. render tsdf volume using open3d
            # use open3d depth renderer
            # verts, faces, norms, colors = tsdf_volume.get_mesh()
            # partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
            # partial_tsdf.export("mesh.ply")
            # mesh = o3d.io.read_triangle_mesh("mesh.ply")
            # depth1, _ = mesh_renderer.render(mesh, curr_pose.cpu().numpy())
            # depth1[depth1 >= far] = -1.
            # depth1 = torch.from_numpy(depth1).to(device)
            # T10 = icp_tracker(depth0, depth1, K)
            # curr_pose = curr_pose @ T10

            # 3. render multi-res tsdf volume
            # dpt1_pyr, _, _, vtx1_pyr, nrm1_pyr, msk1_pyr = tsdf_volume.render_pyramid(curr_pose, K, H, W, n_pyr=4, near=near, far=far)
            # T10 = icp_tracker(depth0, depth1, K, dpt1_pyr=dpt1_pyr, vtx1_pyr=vtx1_pyr, nrm1_pyr=nrm1_pyr, msk1_pyr=msk1_pyr)
            # curr_pose = curr_pose @ T10

        # fusion
        tsdf_volume.integrate(depth0,
                              K,
                              curr_pose,
                              obs_weight=1.,
                              color_img=color0,
                              feat_img=None
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

