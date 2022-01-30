import os
import numpy as np
from data_slam.TUM_RGBD import get_calib

if __name__ == "__main__":
    proc_dir = "/home/jingwen/data/tum_rgbd/rgbd_dataset_freiburg1_rpy/processed"
    K = np.eye(3)
    intri = get_calib()["fr1"]
    K[0, 0] = intri[0]
    K[1, 1] = intri[1]
    K[0, 2] = intri[2]
    K[1, 2] = intri[3]

    camera_dict = np.load(os.path.join(proc_dir, "raw_poses.npz"))
    poses = camera_dict["c2w_mats"]
    P_mats = []
    for c2w in poses:
        w2c = np.linalg.inv(c2w)
        P = K @ w2c[:3, :]
        P_mats += [P]
    np.savez(os.path.join(proc_dir, "cameras.npz"), world_mats=P_mats)