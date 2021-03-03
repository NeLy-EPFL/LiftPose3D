import numpy as np
from liftpose.vision_3d import project_to_random_eangle, normalize_bone_length, project_to_camera
from liftpose.preprocess import anchor_to_root, remove_roots, weird_division, pose_norm
from liftpose.lifter.utils import get_coords_in_dim
import torch


def random_project(eangles, axsorder, vis=None, tvec=None, intr=None):
    def random_project_dispatch(
        inputs, outputs, outputs_raw, keys, stats, roots, target_sets,
    ):
        outputs = outputs_raw.cpu().data.numpy()
        
        # select a camera to project
        if len(eangles)>1:
            whichcam = keys[-1]
            eangle = eangles[whichcam]
            if (tvec is not None) & (intr is not None):
                _tvec = tvec[whichcam]
                _intr = intr[whichcam]
            else:
                _tvec = tvec
                _intr = intr
        else:
            eangle = eangles[0]
            _tvec = tvec
            _intr = intr

        # do random projection
        inputs, _ = project_to_random_eangle(
            outputs[None, :].copy(), eangle, axsorder, project=True, tvec=_tvec, intr=_intr
        )
        
        if len(eangles)>1:
            #zero invisible points
            if vis is not None:
                ind = np.array(vis[whichcam]).astype(bool)
                inputs[:,~ind,:] = 0
        
        #inputs = np.squeeze(inputs)
        inputs = inputs.reshape((1, inputs.size))
        outputs = outputs.reshape((1, outputs.size))

        # anchor points to body-coxa (to predict legjoints wrt body-coxas)
        inputs, _ = anchor_to_root({"inputs": inputs}, roots, target_sets, 2)
        outputs, _ = anchor_to_root({"outputs": outputs}, roots, target_sets, 3)
        
        #normalize pose
        inputs = pose_norm(inputs)

        inputs = inputs["inputs"]
        outputs = outputs["outputs"]

        # Standardize each dimension independently
        np.seterr(divide="ignore", invalid="ignore")
        inputs -= stats["mean_2d"]
        inputs = weird_division(inputs, stats["std_2d"])
        outputs -= stats["mean_3d"]
        outputs = weird_division(outputs, stats["std_3d"])

        # remove roots
        inputs, _ = remove_roots({"inputs": inputs}, target_sets, 2)
        outputs, _ = remove_roots({"outputs": outputs}, target_sets, 3)

        inputs = inputs["inputs"]
        outputs = outputs["outputs"]

        # get torch tensors
        inputs = torch.from_numpy(inputs[0, :]).float()
        outputs = torch.from_numpy(outputs[0, :]).float()

        return inputs, outputs

    return random_project_dispatch


def project_to_cam():
    def project_to_cam_dispatch(
        inputs, outputs, outputs_raw, keys, stats, roots, target_sets
        ):
        
        outputs = outputs_raw.cpu().data.numpy()
        
        inputs = project_to_camera(outputs[None, :], intr=None)
        
        #inputs = np.squeeze(inputs)
        inputs = inputs.reshape((1, inputs.size))
        outputs = outputs.reshape((1, outputs.size))

        # anchor points to body-coxa (to predict legjoints wrt body-coxas)
        inputs, _ = anchor_to_root({"inputs": inputs}, roots, target_sets, 2)
        outputs, _ = anchor_to_root({"outputs": outputs}, roots, target_sets, 3)
        
        #normalize pose
        inputs = pose_norm(inputs)

        inputs = inputs["inputs"]
        outputs = outputs["outputs"]

        # Standardize each dimension independently
        np.seterr(divide="ignore", invalid="ignore")
        inputs -= stats["mean_2d"]
        inputs = weird_division(inputs, stats["std_2d"])
        outputs -= stats["mean_3d"]
        outputs = weird_division(outputs, stats["std_3d"])

        # remove roots
        inputs, _ = remove_roots({"inputs": inputs}, target_sets, 2)
        outputs, _ = remove_roots({"outputs": outputs}, target_sets, 3)

        inputs = inputs["inputs"]
        outputs = outputs["outputs"]

        # get torch tensors
        inputs = torch.from_numpy(inputs[0, :]).float()
        outputs = torch.from_numpy(outputs[0, :]).float()
        
        return inputs, outputs
    
    return project_to_cam_dispatch


def add_noise(noise_amplitude):
    def add_noise_dispatch(
        inputs, outputs, outputs_raw, keys, stats, roots, target_sets
        ):
        """ Add Gaussian noise during training. See above for arguments."""
        targets_2d = get_coords_in_dim(target_sets, 2)
        std = stats["std_2d"][targets_2d]
        std = torch.from_numpy(weird_division(noise_amplitude, std)).float()
        inputs += torch.normal(0, std=std) 

        return inputs, outputs

    return add_noise_dispatch


def perturb_pose(perturb, child, bones, avg_bone_len, std_bone_len):
    def perturb_pose_dispatch(
        inputs, outputs, outputs_raw, keys, stats, roots, target_sets
        ):
        """Perturb pose around mean pose."""

        outputs = outputs_raw.cpu().data.numpy()
        
        #sample ranadom bone lenght around mean
        mean = torch.from_numpy(avg_bone_len).float()
        std = torch.from_numpy(std_bone_len).float()
        bone_length = torch.normal(mean=mean, std=std*perturb).cpu().data.numpy()
        
        # bone_length = calculate_bone_length(outputs[None, :], bones)
        # bone_length *= max(normal.Normal(1,perturb).sample().numpy(),0)
        bone_length = {tuple(bone): bone_length[i] for i,bone in enumerate(bones)}
        
        outputs = normalize_bone_length(outputs[None, :].copy(), 
                                        root=roots[0], 
                                        child=child, 
                                        bone_length=bone_length, 
                                        thr=10)
        
        outputs = outputs.reshape((1, outputs.size))
        
        # get torch tensors
        outputs = torch.from_numpy(outputs[0, :]).float()
        
        return inputs, outputs

    return perturb_pose_dispatch

