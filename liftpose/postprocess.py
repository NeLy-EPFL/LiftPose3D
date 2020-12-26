import torch
import numpy as np
from liftpose.vision_3d import camera_to_world
from liftpose.preprocess import unNormalize, add_roots


def load_test_results(data_dir):
    # load stats
    data = torch.load(data_dir + "/test_results.pth.tar")
    stat_2d, stat_3d = (
        torch.load(data_dir + "/stat_2d.pth.tar"),
        torch.load(data_dir + "/stat_3d.pth.tar"),
    )
    inp_mean = stat_2d["mean"]
    inp_std = stat_2d["std"]
    tar_mean = stat_3d["mean"]
    tar_std = stat_3d["std"]
    targets_2d = stat_2d["targets_2d"]
    targets_3d = stat_3d["targets_3d"]
    offset = stat_3d["offset"]
    offset = np.concatenate([v for k, v in offset.items()], 0)
    inp_offset = np.vstack(list(stat_2d["offset"].values()))
    good_keypts = add_roots(data["good_keypts"], targets_3d, len(tar_mean))

    # TODO make this more universal!!
    if False:
        if np.sum(good_keypts[0, :15]) > 10:
            offset = np.hstack((offset[0, :15], offset[0, :15]))
        else:
            offset = np.hstack((offset[0, 15:], offset[0, 15:]))

    # load predictions
    tar_ = data["target"]
    out_ = data["output"]
    inp_ = data["input"]

    # expand
    tar_ = add_roots(tar_, targets_3d, len(tar_mean))
    out_ = add_roots(out_, targets_3d, len(tar_mean))
    inp_ = add_roots(inp_, targets_2d, len(inp_mean))

    # unnormalise
    tar_ = unNormalize(tar_, tar_mean, tar_std)
    out_ = unNormalize(out_, tar_mean, tar_std)
    inp_ = unNormalize(inp_, inp_mean, inp_std)

    # translate legs back to their original places
    tar_ += offset
    out_ += offset
    inp_ += inp_offset

    return tar_, out_, inp_

