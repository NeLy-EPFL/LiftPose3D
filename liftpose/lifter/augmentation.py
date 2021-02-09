import numpy as np
from liftpose.vision_3d import project_to_random_eangle
from liftpose.preprocess import anchor_to_root, remove_roots
import torch


# TODO not finished
def random_project(eangle, axsorder):
    def random_project_dispatch(
        inputs,
        outputs,
        eangle,
        axsorder,
        stats,
        tvec,
        intr,
        roots,
        target_sets,
        targets_2d,
    ):
        
        raise NotImplementedError

        # do random projection
        inputs = project_to_random_eangle(outputs[None, :], eangle, axsorder, project=True, intr=intr)

        # anchor points to body-coxa (to predict legjoints wrt body-coxas)
        inputs, _ = anchor_to_root({"inputs": inputs}, roots, target_sets, 2)
        outputs, _ = anchor_to_root({"outputs": outputs}, roots, target_sets, 3)

        inputs = inputs["inputs"]
        outputs = outputs["outputs"]

        # Standardize each dimension independently
        np.seterr(divide="ignore", invalid="ignore")
        inputs -= stats['mean_2d']
        inputs /= stats['std_2d']
        outputs -= stats['mean_3d']
        outputs /= stats['std_3d']

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


def add_noise(noise_amplitude):
    def add_noise_dispatch(
        inputs,
        outputs,
        axsorder,
        stats,
        roots,
        target_sets,
        targets_2d,
    ):
        std = stats['std_2d'][targets_2d]
        inputs += torch.from_numpy(
            np.random.normal(0, noise_amplitude / std, size=inputs.shape)
        ).float()

        return inputs, outputs

    return add_noise_dispatch

