import torch
import numpy as np
from liftpose.vision_3d import camera_to_world
from liftpose.preprocess import unNormalize, add_roots


def load_test_results(
    data: dict, stat_2d: dict, stat_3d: dict, prism=False
) -> (np.array, np.array):
    """Transforms vectorized and raw liftpose3d results into [T J 3] format.
        In case out_dim=
        Args:
            data:   
            stat_2d:
            stat_3d:

        Returns:


    """
    inp_mean = stat_2d["mean"]
    inp_std = stat_2d["std"]
    tar_mean = stat_3d["mean"]
    tar_std = stat_3d["std"]
    tar_mean[np.isnan(tar_mean)] = 0
    tar_std[np.isnan(tar_std)] = 1
    targets_2d = stat_2d["targets_2d"]
    targets_3d = stat_3d["targets_3d"]
    offset = stat_3d["offset"]
    offset = np.concatenate([v for k, v in offset.items()], 0)
    inp_offset = np.vstack(list(stat_2d["offset"].values()))
    # assuming root joints are always good_keypts
    good_keypts = add_roots(data["good_keypts"], targets_3d, len(tar_mean), base="ones")

    # TODO make this more universal
    if prism:
        if np.sum(good_keypts[0, :15]) > 10:
            offset = np.hstack((offset[0, :15], offset[0, :15]))
        else:
            offset = np.hstack((offset[0, 15:], offset[0, 15:]))

    # load predictions
    tar = data["target"]
    out = data["output"]
    inp = data["input"]

    # expand
    tar = add_roots(tar, targets_3d, len(tar_mean))
    out = add_roots(out, targets_3d, len(tar_mean))
    inp = add_roots(inp, targets_2d, len(inp_mean))
    print(out[:, 2])
    # unnormalise
    tar = unNormalize(tar, tar_mean, tar_std)
    out = unNormalize(out, tar_mean, tar_std)
    inp = unNormalize(inp, inp_mean, inp_std)
    print(out[:, 2])

    # translate legs back to their original places
    tar += offset
    out += offset
    inp += inp_offset
    print(out[:, 2])
    assert stat_2d["in_dim"] == 2
    inp = inp.reshape(inp.shape[0], -1, 2)
    print(inp[:, 2])
    if stat_3d["out_dim"] == 1:
        out = np.concatenate([inp, out[:, :, np.newaxis]], axis=2)
        tar = np.concatenate([inp, tar[:, :, np.newaxis]], axis=2)
    elif stat_3d["out_dim"] == 3:
        out = out.reshape(out.shape[0], -1, 3)
        tar = tar.reshape(tar.shape[0], -1, 3)
    else:
        raise NotImplementedError

    good_keypts = good_keypts.reshape((good_keypts.shape[0], -1, stat_3d["out_dim"]))
    print(out[:, 2])
    return tar, out, good_keypts

