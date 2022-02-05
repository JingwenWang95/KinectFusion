import time
import torch
import numpy as np
import yaml
from addict import Dict
import argparse


class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)


def load_yaml(path):
    with open(path, encoding='utf8') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    return config


def load_config(args):
    config_dict = load_yaml(args.config)
    # merge args and config
    other_dict = vars(args)
    config_dict.update(other_dict)
    return config_dict


def get_volume_setting(args):
    voxel_size = args.voxel_size
    vol_bnds = np.array(args.vol_bounds).reshape(3, 2)
    vol_dims = (vol_bnds[:, 1] - vol_bnds[:, 0]) // voxel_size + 1
    vol_origin = vol_bnds[:, 0]
    return vol_dims, vol_origin, voxel_size


def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()
