import numpy as np
import torch
import os
from liftpose.vision_3d import camera_to_world
from liftpose.preprocess import unNormalize, add_roots, concat_dict


def load_test_results(
    out_dir: str, stat_2d: dict=None, stat_3d: dict=None, prism=False, cam_par=None
) -> (np.array, np.array):
    """Transforms vectorized and raw liftpose3d results into [T J 3] format.
        In case out_dim=
        Args:
            data:   
            stat_2d:
            stat_3d:

        Returns:


    """
    if stat_2d is None or stat_3d is None:
        stat_2d, stat_3d = (
            torch.load(os.path.join(out_dir, "stat_2d.pth.tar")),
            torch.load(os.path.join(out_dir, "stat_3d.pth.tar")),
        )
    
    inp_mean = stat_2d["mean"]
    inp_std = stat_2d["std"]
    tar_mean = stat_3d["mean"]
    tar_std = stat_3d["std"]
    tar_mean[np.isnan(tar_mean)] = 0
    tar_std[np.isnan(tar_std)] = 1
    targets_2d = stat_2d["targets_2d"]
    targets_3d = stat_3d["targets_3d"]
    offset = stat_3d["offset"]
    offset = concat_dict(offset)
    inp_offset = np.vstack(list(stat_2d["offset"].values()))

    # load predictions
    data = torch.load(os.path.join(out_dir, "test_results.pth.tar"))
    tar = data["target"]
    out = data["output"]
    inp = data["input"]

    # add back roots
    good_keypts = add_roots(data["good_keypts"], targets_3d, len(tar_mean), base="zeros")
    tar = add_roots(tar, targets_3d, len(tar_mean))
    out = add_roots(out, targets_3d, len(tar_mean))
    inp = add_roots(inp, targets_2d, len(inp_mean))
    
    # unnormalise
    tar = unNormalize(tar, tar_mean, tar_std)
    out = unNormalize(out, tar_mean, tar_std)
    inp = unNormalize(inp, inp_mean, inp_std)

    # translate legs back to their original places
    # TODO make this more universal
    if prism:
        if np.sum(good_keypts[0, :15]) > 10:
            offset = np.hstack((offset[0, :15], offset[0, :15]))
        else:
            offset = np.hstack((offset[0, 15:], offset[0, 15:]))
            
    tar += offset
    out += offset
    inp += inp_offset
    
    assert stat_2d["in_dim"] == 2
    inp = inp.reshape(inp.shape[0], -1, 2)
    if stat_3d["out_dim"] == 1:
        out = np.concatenate([inp, out[:, :, np.newaxis]], axis=2)
        tar = np.concatenate([inp, tar[:, :, np.newaxis]], axis=2)
    elif stat_3d["out_dim"] == 3:
        out = out.reshape(out.shape[0], -1, 3)
        tar = tar.reshape(tar.shape[0], -1, 3)
        good_keypts = good_keypts.reshape(tar.shape[0], -1, 3)
    else:
        raise NotImplementedError

    # good_keypts = good_keypts.reshape((good_keypts.shape[0], -1, stat_3d["out_dim"]))
    
    # if cam_par is not None:
    #     tar = camera_to_world(tar, )
    
    return tar, out, good_keypts.astype(bool)