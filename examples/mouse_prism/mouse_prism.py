from load import *
import yaml
import logging
from imp import reload
import matplotlib.pyplot as plt
import load
import os
from liftpose.vision_3d import XY_coord, Z_coord

reload(logging)
logger = logging.getLogger(__name__).setLevel(logging.INFO)

# decleare data parameters
par_train = {
    "data_dir": "/data/LiftPose3D/mouse_prism",  # change the path
    "out_dir": "./out",
    "train_subjects": ["G6AE1", "G6AE2", "G6AE3", "G6AE5"],
    "test_subjects": ["G6AE6"],
    "actions": ["control"],
}

# merge with training parameters
print(os.getcwd())
par_data = yaml.full_load(open("./examples/mouse_prism_package/param.yaml", "rb"))
par = {**par_data["data"], **par_train}

# Load data
train, train_keypts, _ = load.load_3D(
    par["data_dir"], subjects=par["train_subjects"], actions=par["actions"]
)
test, test_keypts, _ = load.load_3D(
    par["data_dir"], subjects=par["test_subjects"], actions=par["actions"]
)


train_2d, train_3d = XY_coord(train), Z_coord(train)
test_2d, test_3d = XY_coord(test), Z_coord(test)

from liftpose.main import train as lp3d_train

lp3d_train(
    train_2d=train_2d,
    test_2d=test_2d,
    train_3d=train_3d,
    test_3d=test_3d,
    train_keypts=train_keypts,
    test_keypts=test_keypts,
    roots=par["roots"],
    target_sets=par["targets"],
    in_dim=2,
    out_dim=par["out_dim"],
    out_dir=par["out_dir"],
)

