import os
import time
import argparse
import numpy as np
import torch
import open3d as o3d
from fusion import TSDFVolumeTorch
from dataset.tum_rgbd import TUMDataset
from tracker import ICPTracker
from utils import load_config, get_volume_setting, get_time


vis_param = argparse.Namespace()
vis_param.frame_id = 0
vis_param.current_mesh = None
vis_param.current_camera = None
vis_param.curr_pose = None


def refresh(vis):
    if vis:
        # This spares slots for meshing thread to emit commands.
        time.sleep(0.01)

    if vis_param.frame_id == vis_param.n_frames:
        return False

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
    # update view-point
    if vis_param.args.follow_camera:
        follow_camera(vis, vis_param.curr_pose.cpu().numpy())
    # fusion
    vis_param.map.integrate(depth0, K, vis_param.curr_pose, obs_weight=1., color_img=color0)
    # update mesh
    mesh = vis_param.map.to_o3d_mesh()
    if vis_param.current_mesh is not None:
        vis.remove_geometry(vis_param.current_mesh, reset_bounding_box=False)
    vis.add_geometry(mesh, reset_bounding_box=False)
    vis_param.current_mesh = mesh
    # update camera
    camera = draw_camera(vis_param.curr_pose.cpu().numpy())
    if vis_param.current_camera is not None:
        vis.remove_geometry(vis_param.current_camera, reset_bounding_box=False)
    vis.add_geometry(camera, reset_bounding_box=False)
    vis_param.current_camera = camera

    vis_param.frame_id += 1
    return True


def draw_camera(c2w, cam_width=0.2, cam_height=0.15, f=0.1):
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [[1, 0, 1] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    return line_set


def follow_camera(vis, c2w, z_offset=-2):
    """
    :param vis: visualizer handle
    :param c2w: world to camera transform Twc
    :param z_offset: offset along z-direction of eye wrt camera
    :return:
    """
    e2c = np.eye(4)
    e2c[2, 3] = z_offset
    e2w = c2w @ e2c
    set_view(vis, np.linalg.inv(e2w))


def set_view(vis, w2e=np.eye(4)):
    """
    :param vis: visualizer handle
    :param w2e: world-to-eye transform
    :return:
    """
    vis_ctl = vis.get_view_control()
    cam = vis_ctl.convert_to_pinhole_camera_parameters()
    # world to eye w2e
    cam.extrinsic = w2e
    vis_ctl.convert_from_pinhole_camera_parameters(cam)


def get_view(vis):
    vis_ctl = vis.get_view_control()
    cam = vis_ctl.convert_to_pinhole_camera_parameters()
    print(cam.extrinsic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fr1_desk.yaml", help="Path to config file.")
    parser.add_argument("--follow_camera", action="store_true", help="Make view-point follow the camera motion")
    args = load_config(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    dataset = TUMDataset(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25)
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
    vis.create_window(width=1280, height=960)
    # vis.get_view_control().unset_constant_z_near()
    # vis.get_view_control().unset_constant_z_far()
    vis.get_render_option().mesh_show_back_face = True
    vis.register_animation_callback(callback_func=refresh)
    coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coord_axes)
    vis.remove_geometry(coord_axes, reset_bounding_box=False)
    # set initial view-point
    c2w0 = dataset[0][2]
    follow_camera(vis, c2w0.cpu().numpy())
    # start reconstruction and visualization
    vis.run()
    vis.destroy_window()
