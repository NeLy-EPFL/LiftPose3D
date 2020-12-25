import torch
import numpy as np
from liftpose.vision_3d import camera_to_world
from liftpose.preprocess import unNormalize, add_roots


def load_test_results(data_dir):
    # load stats
    tar_mean = torch.load(data_dir + "/stat_3d.pth.tar")["mean"]
    tar_std = torch.load(data_dir + "/stat_3d.pth.tar")["std"]
    targets_3d = torch.load(data_dir + "/stat_3d.pth.tar")["targets_3d"]
    #cam_par = torch.load(data_dir + "/stat_3d.pth.tar")["rcams"]
    #cam_par = list(cam_par.values())
    offset = torch.load(data_dir + "/stat_3d.pth.tar")["offset"]
    offset = np.concatenate([v for k, v in offset.items()], 0)

    # load predictions
    data = torch.load(data_dir + "/test_results.pth.tar")
    tar_ = data["target"]
    out_ = data["output"]

    # expand
    tar_ = add_roots(tar_, targets_3d, len(tar_mean))
    out_ = add_roots(out_, targets_3d, len(tar_mean))

    # unnormalise
    tar_ = unNormalize(tar_, tar_mean, tar_std)
    out_ = unNormalize(out_, tar_mean, tar_std)

    # translate legs back to their original places
    tar_ += offset[0, :]
    out_ += offset[0, :]

    # transform back to worldcamera_to_world( poses_cam, cam_par, cam )
    # tar_ = camera_to_world(tar_, cam_par[0][cam])
    # out_ = camera_to_world(out_, cam_par[0][cam])

    return tar_, out_

