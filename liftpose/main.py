import glob
import logging
import os
import sys
from pprint import pformat

import numpy as np
import torch

from liftpose.lifter.lift import network_main
from liftpose.lifter.opt import Options
from liftpose.preprocess import preprocess_2d, preprocess_3d, init_keypts, flatten_dict

# deterministic training
torch.manual_seed(0)
np.random.seed(0)


# set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
)


def train_np(
    train_2d: np.ndarray,
    test_2d: np.ndarray,
    train_3d: np.ndarray,
    test_3d: np.ndarray,
    out_dir: str,
    root: int = 0,
    train_keypts: dict = None,
    test_keypts: dict = None,
) -> None:
    n_joints = train_2d.shape[1]
    k = ""
    train_2d = {k: train_2d}
    train_3d = {k: train_3d}
    test_2d = {k: test_2d}
    test_3d = {k: test_3d}

    roots = [root]
    target_sets = list(
        set(range(n_joints)) - set(roots)
    )  # every point except the root is going to be predicted
    target_sets = [target_sets]

    if train_keypts is not None:
        train_keypts = {k: train_keypts}
    if test_keypts is not None:
        test_keypts = {k: test_keypts}

    train(
        train_2d=train_2d,
        test_2d=test_2d,
        train_3d=train_3d,
        test_3d=test_3d,
        roots=roots,
        target_sets=target_sets,
        out_dir=out_dir,
        train_keypts=train_keypts,
        test_keypts=test_keypts,
    )


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

    """Train LiftPose3D.
        Training works in two steps:
        1. Training and testing data will be preprocessed to zero-mean and unit-std. 
            Root joints will be removed from the training and target_set will be subtracted from the root joints.  
            Corresponding normalization statistics will be written under stat_2d and stat_3d files.
            Normalized 2d and 3d data will written under train_2d and train_3d files.
        2. Martinez et. al network will be trained on the processed training data using liftpose.lifter module.

    Args:
        train_2d: Dict[Tuple:np.array[float]]
            Dictionary with keys as experiment names and values as numpy arrays in the shape of [T J 2]. 
            T corresponds to time axis and J is number of joints. T can be arbitrary.
            The last dimension 2 corresponds to the 2d pose. 
        test_2d: Dict[Tuple:np.array[float]]
            Dictionary with keys as experiment names and values as numpy arrays in the shape of [T J 2]. 
            T corresponds to time axis and J is number of joints. T can be arbitrary.
            The last dimension 2 corresponds to the 2d pose.  
        train_3d: Dict[Tuple:np.array[float]]
            Dictionary with keys as experiment names and values as numpy arrays in the shape of [T J out_dim]. 
            T corresponds to time axis and J is number of joints. T can be arbitrary.
            out_dim can only be 1 or 3. 
        test_3d: Dict[Tuple:np.array[float]] 
            Dictionary with keys as experiment names and values as numpy arrays in the shape of [T J out_dim]. 
            T corresponds to time axis and J is number of joints. T can be arbitrary.
            out_dim can only be 1 or 3. 
        roots: List[Int]
            Single depth list consisting of root joints. Corresponding
            target set will predicted with respect to the root joint. 
            Cannot be empty.
        target_sets: List[List[Int]]
            Joints to be predicted with respect to roots.
            if roots = [0, 1] and target_sets = [[2,3], [4,5]], then the
            network will predict the relative location Joint 2 and 3 with respect to Joint 0.
            Likewise Joint location 4 and 5 will be predicted with respect to Joint 1.
            Cannot be empty.
        out_dir: String
            **relative** output path, will be created if does exist. 
        train_keypts: Dict[Tuple:np.array[bool]]
            Dictionary with same keys as train_3d. Values should have the same shape as train_3d, however instead should be boolean arrays.
            A point will not be used during training/testing in case corresponding boolean value is false.
        test_keypts: Dict[Tuple:np.array[bool]]
            Dictionary with same keys as train_3d. Values should have the same shape as train_3d, however instead should be boolean arrays.
            A point will not be used during training/testing in case corresponding boolean value is false.

    Returns:
        None

    This function will create a folder in 
    the relative path of out_dir, if does not exist. 
    Following files will be written under out_dir:
        1. "stat_2d.pth.tar" 
        2. "stat_3d.pth.tar"
        3. "train_2d.pth.tar"
        4. "train_3d.pth.tar"
        5. "test_2d.pth.tar"
        6. "test_3d.pth.tar"
        7. "ckpt_last.pth.tar"
        8. "ckpt_best.pth.tar"
        9. "log_train.txt"
    """
    # fmt: off
    assert all(t.ndim == 3 for t in list(train_2d.values()))
    assert all(t.ndim == 3 for t in list(test_2d.values()))
    assert all(t.ndim == 3 for t in list(train_2d.values()))
    assert all(t.ndim == 3 for t in list(test_3d.values()))
    assert len(set([v.shape[-1] for v in train_2d.values()])) == 1
    assert len(set([v.shape[-1] for v in train_3d.values()])) == 1
    assert len(set([v.shape[-1] for v in test_2d.values()])) == 1
    assert len(set([v.shape[-1] for v in test_3d.values()])) == 1
    assert all([t3d.shape[0] == t2d.shape[0] for (t3d,t2d) in zip(list(train_3d.values()), list(train_2d.values()))])
    assert all([t3d.shape[1] == t2d.shape[1] for (t3d,t2d) in zip(list(train_3d.values()), list(train_2d.values()))])
    assert all([kp.shape == t.shape for (kp,t) in zip(list(test_3d.values()), list(test_keypts.values()))])
    assert len(roots) != 0
    assert len(target_sets) != 0
    assert len(roots) == len(target_sets) # number of root joint and number of targert sets are equivalent

    # create out_dir if it does not exists
    if not os.path.exists(out_dir):
        logger.info(f"Creating directory {os.path.abspath(out_dir)}")
        os.makedirs(out_dir)

    # init default keypts in case it is None
    train_keypts = init_keypts(train_3d) if train_keypts is None else train_keypts
    test_keypts = init_keypts(test_3d) if test_keypts is None else test_keypts
    
    # make sure keypts are in the correct shape
    # keypts should be in the same shape of corresponding train3d and test3d values
    assert all([kp.shape == t.shape for (kp,t) in zip(list(train_3d.values()), list(train_keypts.values()))])
    assert all([kp.shape == t.shape for (kp,t) in zip(list(test_3d.values()), list(test_keypts.values()))])
    
    # init data
    in_dim = list(train_2d.values())[0].shape[-1]
    out_dim = list(train_3d.values())[0].shape[-1]
    assert (out_dim == 1 or out_dim == 3), f"out_dim can only be 1 or 3, wheres set as {out_dim}"
    train_2d, test_2d = flatten_dict(train_2d), flatten_dict(test_2d)
    train_3d, test_3d = flatten_dict(train_3d), flatten_dict(test_3d)    
    # fmt: on

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

    # flatten train_keypts
    train_keypts = flatten_dict(train_keypts)
    test_keypts = flatten_dict(test_keypts)

    train_keypts = {k: v[:, targets_3d] for (k, v) in train_keypts.items()}
    test_keypts = {k: v[:, targets_3d] for (k, v) in test_keypts.items()}

    # save 3d data
    logger.info(
        f'Saving pre-processed 3D data at {os.path.abspath(os.path.join(out_dir + "stat_3d.pth.tar."))}'
    )

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
    logger.info("Starting training model.")

    option = Options().parse()
    option.data_dir = os.path.abspath(out_dir)
    option.out = os.path.abspath(out_dir)
    option.out_dir = os.path.abspath(out_dir)

    logger.debug("\n==================Options=================")
    logger.debug(pformat(vars(option), indent=4))
    logger.debug("==========================================\n")
    network_main(option)


def test(out_dir: str) -> None:
    """Test LiftPose3D.
        Runs pre-trained liftpose3d model (saved as ckpt_best.pth.tar, using the liftpose.main.train function.)
            on the test data (saved as test_2d.ckpt.tar and test_3d.ckpt.tar).
        Train function must be called with the same out_dir before calling liftpose.main.test. Assumes out_dir existss.
        Saves test_results.pth.tar under out_dir folder.

        Args:
            out_dir: the same folder used during training (liftpose.main.train) 
        
        Returns:
            Nones
    """

    assert os.path.isdir(
        out_dir
    ), "{out_dir} does not exists. please call liftpose.main.train function first with the same path."

    logger.info("starting testing in path: {}".format(out_dir))
    option = Options().parse()
    option.data_dir = os.path.abspath(out_dir)
    option.out = os.path.abspath(out_dir)  # TODO do we need to set out?
    option.out_dir = os.path.abspath(out_dir)
    option.test = True
    option.load = glob.glob(os.path.join(out_dir, "ckpt_best.pth.tar"))[0]
    option.predict = False
    network_main(option)
