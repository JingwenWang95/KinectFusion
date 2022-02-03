import os
import time
import argparse
import torch
import open3d as o3d
from fusion import TSDFVolumeTorch
from dataset.TUM_RGBD import TUMDataset, TUMDatasetOnline
from Tracker import ICPTracker
from utils import load_config, get_volume_setting, get_time


vis_param = argparse.Namespace()
vis_param.frame_id = 0
vis_param.current_mesh = None
vis_param.curr_pose = None


def refresh(vis):
    if vis:
        # This spares slots for meshing thread to emit commands.
        time.sleep(0.01)

    sample = vis_param.dataset[vis_param.frame_id]
    color0, depth0, pose_gt, K = sample  # use live image as template image (0)
    # depth0[depth0 <= 0.5] = 0.
    if vis_param.frame_id == 0:
        vis_param.curr_pose = pose_gt
    else:
        # render depth image (1) from tsdf volume
        depth1, color1, vertex01, normal1, mask1 = \
            vis_param.map.render_model(vis_param.curr_pose, K, vis_param.H, vis_param.W,
            near=args.near, far=vis_param.args.far, n_samples=vis_param.args.n_steps)
        T10 = vis_param.tracker(depth0, depth1, K)  # transform from 0 to 1
        vis_param.curr_pose = vis_param.curr_pose @ T10

    # fusion
    vis_param.map.integrate(depth0, K, vis_param.curr_pose, obs_weight=1., color_img=color0)
    mesh = vis_param.map.to_o3d_mesh()
    if vis_param.current_mesh is not None:
        vis.remove_geometry(vis_param.current_mesh, reset_bounding_box=False)

    vis.add_geometry(mesh, reset_bounding_box=False)
    vis_param.current_mesh = mesh
    vis_param.frame_id += 1

    if vis_param.frame_id == vis_param.n_frames:
        return False
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/fr1_room.yaml", help='Path to config file.')
    args = load_config(parser.parse_args())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUMDatasetOnline(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25)
    vol_dims, vol_origin, voxel_size = get_volume_setting(args)

    vis_param.args = args
    vis_param.dataset = dataset
    vis_param.map = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True)
    vis_param.tracker = ICPTracker(args, device)
    vis_param.n_frames = len(dataset)
    vis_param.H = dataset.H
    vis_param.W = dataset.W

    # visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True
    vis.register_animation_callback(callback_func=refresh)
    # vis.register_key_callback(key=ord("."), callback_func=refresh)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    vis.run()
    vis.destroy_window()
