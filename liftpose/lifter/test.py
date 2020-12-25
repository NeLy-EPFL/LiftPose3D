import logging
import sys

import liftpose.lifter.utils as utils
import numpy as np
from liftpose.preprocess import normalize, unNormalize
from torch.autograd import Variable
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
)


def test(test_loader, model, criterion, stat, predict=False):
    losses = utils.AverageMeter()
    model.eval()

    all_dist, all_output, all_target, all_input, all_bool = [], [], [], [], []

    for i, (inps, tars, good_keypts, keys) in enumerate(test_loader):

        # make prediction with model
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))
        outputs = model(inputs)
        all_output.append(outputs.data.cpu().numpy())
        all_input.append(inputs.data.cpu().numpy())

        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        outputs[~good_keypts] = 0
        targets[~good_keypts] = 0

        # undo normalisation to calculate accuracy in real units
        dim = stat["out_dim"]
        dimensions = stat["targets_3d"]
        tar = unNormalize(
            targets.data.cpu().numpy(),
            stat["mean"][dimensions],
            stat["std"][dimensions],
        )
        out = unNormalize(
            outputs.data.cpu().numpy(),
            stat["mean"][dimensions],
            stat["std"][dimensions],
        )

        abserr = np.abs(out - tar)

        # compute error
        n_pts = len(dimensions) // dim
        distance = np.zeros_like(abserr)

        # group and stack
        """
        all_dist.append(distance)
        all_output.append(outputs.data.cpu().numpy())
        all_target.append(targets.data.cpu().numpy())
        all_input.append(inputs.data.cpu().numpy())
        """
        for k in range(n_pts):
            distance[:, k] = np.sqrt(np.sum(abserr[:, dim * k : dim * (k + 1)], axis=1))

        all_dist.append(distance)
        all_target.append(targets.data.cpu().numpy())
        all_bool.append(good_keypts)

    """
    all_dist, all_output, all_target, all_input = (
        np.vstack(all_dist),
        np.vstack(all_output),
        np.vstack(all_target),
        np.vstack(all_input),
    )
    """
    all_input = np.vstack(all_input)
    all_output = np.vstack(all_output)

    if predict:
        return None, None, None, None, all_output, None, all_input, None

    all_target = np.vstack(all_target)
    all_dist = np.vstack(all_dist)
    all_bool = np.vstack(all_bool)

    # mean errors
    all_dist[all_dist == 0] = np.nan
    joint_err = np.nanmean(all_dist, axis=0)
    ttl_err = np.nanmean(joint_err)

    # logger.info("test error: {}".format(ttl_err))
    return (
        losses.avg,
        ttl_err,
        joint_err,
        all_dist,
        all_output,
        all_target,
        all_input,
        all_bool,
    )
