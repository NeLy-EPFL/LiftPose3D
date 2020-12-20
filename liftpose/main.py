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

# from liftpose.vision_3d import transform_frame
# TODO check reprojection error, warn if large
# TODO better docstring
# TODO do we expect 2d or 3d arrays?


def train(
    train_2d: dict,
    test_2d: dict,
    train_3d: dict,
    test_3d: dict,
    rcams_train: dict,
    rcams_test: dict,
    roots: list,
    target_sets: list,
    in_dim: int,
    out_dim: int,
    out_dir: str,
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
    logger = logging.getLogger("lp3d")
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    # TODO do more sanity check on the data
    assert len(roots) == len(
        target_sets
    ), "number of elements in roots and target_sets does not match in params.yaml"
    assert (
        train_3d.keys() == rcams_train.keys()
    ), "keys do nt match between train_3d and rcams_train"
    assert (
        test_3d.keys() == rcams_test.keys()
    ), "keys do not match between test_3d and rcams_test"

    # create out_dir if it does not exists
    if not os.path.exists(out_dir):
        logger.info(f"Creating directory {os.path.abspath(out_dir)}")
        os.makedirs(out_dir)

    # preprocess 2d
    train_set_2d, test_set_2d, mean_2d, std_2d, targets_2d = preprocess_2d(
        train_2d, test_2d, roots, target_sets, in_dim
    )

    # save 2d data
    logger.info(
        f'Saving pre-processed 2D data at {os.path.abspath(out_dir + "/stat_2d.pth.tar.")}'
    )
    torch.save(train_set_2d, out_dir + "/train_2d.pth.tar")
    torch.save(test_set_2d, out_dir + "/test_2d.pth.tar")
    torch.save(
        {"mean": mean_2d, "std": std_2d, "targets_2d": targets_2d},
        out_dir + "/stat_2d.pth.tar",
    )

    # preprocess 3d
    (
        train_set_3d,
        test_set_3d,
        mean_3d,
        std_3d,
        targets_3d,
        rcams_test,
        offset,
    ) = preprocess_3d(
        train_3d, test_3d, roots, target_sets, out_dim, rcams_train, rcams_test
    )

    # save 3d data
    logger.info(
        f'Saving pre-processed 3D data at {os.path.abspath(out_dir + "/stat_3d.pth.tar.")}'
    )
    torch.save(train_set_3d, out_dir + "/train_3d.pth.tar")
    torch.save(test_set_3d, out_dir + "/test_3d.pth.tar")
    torch.save(
        {
            "mean": mean_3d,
            "std": std_3d,
            "targets_3d": targets_3d,
            "rcams": rcams_test,
            "offset": offset,
            "output_size": len(targets_3d),
            "input_size": len(targets_2d),
        },
        out_dir + "/stat_3d.pth.tar",
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
    option = Options().parse()
    option.data_dir = os.path.abspath(out_dir)
    option.out = os.path.abspath(out_dir)  # TODO do we need to set out?
    option.out_dir = os.path.abspath(out_dir)
    option.test = True
    option.load = glob.glob(out_dir + "/ckpt_best.pth.tar")[0]
    network_main(option)
