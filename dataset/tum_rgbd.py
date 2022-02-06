import torch
from os import path
from tqdm import tqdm
import imageio
import cv2
import numpy as np
import open3d as o3d


def get_calib():
    return {
        "fr1": [517.306408, 516.469215, 318.643040, 255.313989],
        "fr2": [520.908620, 521.007327, 325.141442, 249.701764],
        "fr3": [535.4, 539.2, 320.1, 247.6]
    }


# Note,this step converts w2c (Tcw) to c2w (Twc)
def load_K_Rt_from_P(P):
    """
    modified from IDR https://github.com/lioryariv/idr
    """
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # convert from w2c to c2w
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class TUMDataset(torch.utils.data.Dataset):
    """
    TUM dataset loader, pre-load images in advance
    """

    def __init__(
            self,
            rootdir,
            device,
            near: float = 0.2,
            far: float = 5.,
            img_scale: float = 1.,  # image scale factor
            start: int = -1,
            end: int = -1,
    ):
        super().__init__()
        assert path.isdir(rootdir), f"'{rootdir}' is not a directory"
        self.device = device
        self.c2w_all = []
        self.K_all = []
        self.rgb_all = []
        self.depth_all = []

        # root should be tum_sequence
        data_path = path.join(rootdir, "processed")
        cam_file = path.join(data_path, "cameras.npz")
        print("LOAD DATA", data_path)

        # world_mats, normalize_mat
        cam_dict = np.load(cam_file)
        world_mats = cam_dict["world_mats"]  # K @ w2c

        d_min = []
        d_max = []
        # TUM saves camera poses in OpenCV convention
        for i, world_mat in enumerate(tqdm(world_mats)):
            # ignore all the frames betfore
            if start > 0 and i < start:
                continue
            # ignore all the frames after
            if 0 < end < i:
                break

            intrinsics, c2w = load_K_Rt_from_P(world_mat)
            c2w = torch.tensor(c2w, dtype=torch.float32)
            # read images
            rgb = np.array(imageio.imread(path.join(data_path, "rgb/{:04d}.png".format(i)))).astype(np.float32)
            depth = np.array(imageio.imread(path.join(data_path, "depth/{:04d}.png".format(i)))).astype(np.float32)
            depth /= 5000.  # TODO: put depth factor to args
            d_max += [depth.max()]
            d_min += [depth.min()]
            # depth = cv2.bilateralFilter(depth, 5, 0.2, 15)
            # print(depth[depth > 0.].min())
            invalid = (depth < near) | (depth > far)
            depth[invalid] = -1.
            # downscale the image size if needed
            if img_scale < 1.0:
                full_size = list(rgb.shape[:2])
                rsz_h, rsz_w = [round(hw * img_scale) for hw in full_size]
                # TODO: figure out which way is better: skimage.rescale or cv2.resize
                rgb = cv2.resize(rgb, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (rsz_w, rsz_h), interpolation=cv2.INTER_NEAREST)
                intrinsics[0, 0] *= img_scale
                intrinsics[1, 1] *= img_scale
                intrinsics[0, 2] *= img_scale
                intrinsics[1, 2] *= img_scale

            self.c2w_all.append(c2w)
            self.K_all.append(torch.from_numpy(intrinsics[:3, :3]))
            self.rgb_all.append(torch.from_numpy(rgb))
            self.depth_all.append(torch.from_numpy(depth))
        print("Depth min: {:f}".format(np.array(d_min).min()))
        print("Depth max: {:f}".format(np.array(d_max).max()))
        self.n_images = len(self.rgb_all)
        self.H, self.W, _ = self.rgb_all[0].shape

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        return self.rgb_all[idx].to(self.device), self.depth_all[idx].to(self.device), \
               self.c2w_all[idx].to(self.device), self.K_all[idx].to(self.device)


class TUMDatasetOnline(torch.utils.data.Dataset):
    """
    Online TUM dataset loader, load images when __getitem__() is called
    """

    def __init__(
            self,
            rootdir,
            device,
            near: float = 0.2,
            far: float = 5.,
            img_scale: float = 1.,  # image scale factor
            start: int = -1,
            end: int = -1,
    ):
        super().__init__()
        assert path.isdir(rootdir), f"'{rootdir}' is not a directory"
        self.device = device
        self.img_scale = img_scale
        self.near = near
        self.far = far
        self.c2w_all = []
        self.K_all = []
        self.rgb_files_all = []
        self.depth_files_all = []

        # root should be tum_sequence
        data_path = path.join(rootdir, "processed")
        cam_file = path.join(data_path, "cameras.npz")
        print("LOAD DATA", data_path)

        # world_mats, normalize_mat
        cam_dict = np.load(cam_file)
        world_mats = cam_dict["world_mats"]  # K @ w2c

        # TUM saves camera poses in OpenCV convention
        for i, world_mat in enumerate(world_mats):
            # ignore all the frames betfore
            if start > 0 and i < start:
                continue
            # ignore all the frames after
            if 0 < end < i:
                break

            intrinsics, c2w = load_K_Rt_from_P(world_mat)
            c2w = torch.tensor(c2w, dtype=torch.float32)
            self.c2w_all.append(c2w)
            self.K_all.append(torch.from_numpy(intrinsics[:3, :3]))
            self.rgb_files_all.append(path.join(data_path, "rgb/{:04d}.png".format(i)))
            self.depth_files_all.append(path.join(data_path, "depth/{:04d}.png".format(i)))

        self.n_images = len(self.rgb_files_all)
        H, W, _ = np.array(imageio.imread(self.rgb_files_all[0])).shape
        self.H = round(H * img_scale)
        self.W = round(W * img_scale)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        K = self.K_all[idx].to(self.device)
        c2w = self.c2w_all[idx].to(self.device)
        # read images
        rgb = np.array(imageio.imread(self.rgb_files_all[idx])).astype(np.float32)
        depth = np.array(imageio.imread(self.depth_files_all[idx])).astype(np.float32)
        depth /= 5000.
        # depth = cv2.bilateralFilter(depth, 5, 0.2, 15)
        depth[depth < self.near] = 0.
        depth[depth > self.far] = -1.
        # downscale the image size if needed
        if self.img_scale < 1.0:
            full_size = list(rgb.shape[:2])
            rsz_h, rsz_w = [round(hw * self.img_scale) for hw in full_size]
            rgb = cv2.resize(rgb, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (rsz_w, rsz_h), interpolation=cv2.INTER_NEAREST)
            K[0, 0] *= self.img_scale
            K[1, 1] *= self.img_scale
            K[0, 2] *= self.img_scale
            K[1, 2] *= self.img_scale

        rgb = torch.from_numpy(rgb).to(self.device)
        depth = torch.from_numpy(depth).to(self.device)

        return rgb, depth, c2w, K

