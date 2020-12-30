import numpy as np
import torch
import os
import logging
import sys
from pprint import pformat
import glob
from liftpose.lifter.opt import Options
from liftpose.lifter.lift import network_main

from liftpose.preprocess import preprocess_2d, preprocess_3d


logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
)


def init_keypts(train_3d):
    # TODO why hard-code 2?
    return {
        k: np.ones((v.shape[0], v.shape[1] * 2), dtype=np.bool)
        for (k, v) in train_3d.items()
    }


def flatten_dict(d):
    for (k, v) in d.items():
        d[k] = v.reshape(v.shape[0], v.shape[1] * v.shape[2])
    return d


def train(
    train_2d: dict,
    test_2d: dict,
    train_3d: dict,
    test_3d: dict,
    roots: list,
    target_sets: list,
    out_dir: str,
    train_keypts: dict = None,
    test_keypts: dict = None,
) -> None:

    """
    Train LiftPose model

    Args
        train_2d: dictionary with 3d poses in world coordinates
        train_3d: dictionary with camera parameters
        roots: camera_id to consider
        out_dir: output path, where indermeadiate and final files will be saved 
    Returns
        None
    """

    # init data
    in_dim = list(train_2d.values())[0].shape[-1]
    out_dim = list(train_3d.values())[0].shape[-1]
    train_2d, test_2d = flatten_dict(train_2d), flatten_dict(test_2d)
    train_3d, test_3d = flatten_dict(train_3d), flatten_dict(test_3d)

    if train_keypts is None:
        train_keypts = init_keypts(train_2d)
    if test_keypts is None:
        test_keypts = init_keypts(test_2d)

    # TODO assert dimensionality is uniform in data
    assert len(roots) == len(
        target_sets
    ), "number of elements in roots and target_sets does not match in params.yaml"

    # create out_dir if it does not exists
    if not os.path.exists(out_dir):
        logger.info(f"Creating directory {os.path.abspath(out_dir)}")
        os.makedirs(out_dir)

    # preprocess 2d
    train_set_2d, test_set_2d, mean_2d, std_2d, targets_2d, offset_2d = preprocess_2d(
        train_2d, test_2d, roots, target_sets, in_dim
    )

    # save 2d data
    logger.info(
        f'Saving pre-processed 2D data at {os.path.abspath(os.path.join(out_dir, "stat_2d.pth.tar."))}'
    )
    torch.save(train_set_2d, os.path.join(out_dir, "train_2d.pth.tar"))
    torch.save(test_set_2d, os.path.join(out_dir, "test_2d.pth.tar"))
    torch.save(
        {
            "mean": mean_2d,
            "std": std_2d,
            "in_dim": in_dim,
            "targets_2d": targets_2d,
            "offset": offset_2d,
        },
        os.path.join(out_dir, "stat_2d.pth.tar"),
    )

    # preprocess 3d
    (train_set_3d, test_set_3d, mean_3d, std_3d, targets_3d, offset,) = preprocess_3d(
        train_3d, test_3d, roots, target_sets, out_dim
    )

    # save 3d data
    logger.info(
        f'Saving pre-processed 3D data at {os.path.abspath(os.path.join(out_dir + "stat_3d.pth.tar."))}'
    )

    # TODO not clear what this does
    for key in train_keypts.keys():
        train_keypts[key] = train_keypts[key][:, targets_3d]
    for key in test_keypts.keys():
        test_keypts[key] = test_keypts[key][:, targets_3d]

    torch.save([train_set_3d, train_keypts], os.path.join(out_dir, "train_3d.pth.tar"))
    torch.save([test_set_3d, test_keypts], os.path.join(out_dir, "test_3d.pth.tar"))
    torch.save(
        {
            "mean": mean_3d,
            "std": std_3d,
            "targets_3d": targets_3d,
            "offset": offset,
            "LR_train": train_keypts,
            "LR_test": test_keypts,
            "out_dim": out_dim,
            "input_size": len(targets_2d),
            "output_size": len(targets_3d),
        },
        os.path.join(out_dir, "stat_3d.pth.tar"),
    )

    # Starting to train Martinez et. al model
    logger.info("Starting training model")

    # TODO bit hacky, we should get inputs for the network from another yaml file
    # TODO also everything should be explicit function argument, instead of be hidden in dictionary
    option = Options().parse()
    option.data_dir = os.path.abspath(out_dir)
    option.out = os.path.abspath(out_dir)  # TODO do we need to set out?
    option.out_dir = os.path.abspath(out_dir)

    logger.debug("\n==================Options=================")
    logger.debug(pformat(vars(option), indent=4))
    logger.debug("==========================================\n")
    network_main(option)


def test(out_dir: str):
    logger.info("starting testing in path: {}".format(out_dir))
    option = Options().parse()
    option.data_dir = os.path.abspath(out_dir)
    option.out = os.path.abspath(out_dir)  # TODO do we need to set out?
    option.out_dir = os.path.abspath(out_dir)
    option.test = True
    option.load = glob.glob(os.path.join(out_dir, "ckpt_best.pth.tar"))[0]
    option.predict = True
    network_main(option)
