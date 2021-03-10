from load import *
import torch
import yaml
import logging
from imp import reload
import matplotlib.pyplot as plt
from liftpose.vision_3d import world_to_camera_dict, reprojection_error
reload(logging)
logger = logging.getLogger(__name__).setLevel(logging.INFO)
from tqdm import tqdm
tqdm.get_lock().locks = []


for cam in [[1], [2], [5]]:
    # decleare data parameters
    par_train = {  'data_dir'       : '/data/LiftPose3D_2602/fly_tether/data_DF3D/', # change the path
                   'out_dir'        : 'out_drop',
                   'train_subjects' : [1,2,3,4,5],
                   'test_subjects'  : [6,7],
                   'actions'        : ['all'],
                   'cam_id'         : [2]}

    par_train['cam_id'] = cam
    par_train['out_dir'] = f'out_cam{cam}'

    # merge with training parameters
    par_data = yaml.full_load(open('param.yaml', "rb"))
    par = {**par_data["data"], **par_train}
    # Load 2D data
    train_2d = load_2D(
        par["data_dir"],
        par,
        cam_id=par["cam_id"],
        subjects=par["train_subjects"],
        actions=par["actions"],
    )
    test_2d = load_2D(
        par["data_dir"],
        par,
        cam_id=par["cam_id"],
        subjects=par["test_subjects"],
        actions=par["actions"],
    )

    # Load 3D data
    train_3d, train_keypts, rcams_train = load_3D(
        par["data_dir"],
        par,
        cam_id=par["cam_id"],
        subjects=par["train_subjects"],
        actions=par["actions"],
    )
    test_3d, test_keypts, rcams_test = load_3D(
        par["data_dir"],
        par,
        cam_id=par["cam_id"],
        subjects=par["test_subjects"],
        actions=par["actions"],
    )

    train_3d = world_to_camera_dict(train_3d, rcams_train)
    test_3d = world_to_camera_dict(test_3d, rcams_test)

    from liftpose.postprocess import load_test_results
    from liftpose.main import test as lp3d_test
    from liftpose.main import train as lp3d_train

    lp3d_train(train_2d=train_2d, test_2d=test_2d,
               train_3d=train_3d, test_3d=test_3d,
               train_keypts=train_keypts,
               test_keypts=test_keypts,
               roots=par['roots'],
               target_sets=par['target_sets'],
               out_dir=par['out_dir'],
               training_kwargs={"epochs":100, "job":1, "lr_decay":100000, "lr_gamma":0.5, "dropout":0.0, "epochs":30})
    
    
    lp3d_test(par['out_dir'])

