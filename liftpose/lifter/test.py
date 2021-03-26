import logging
import sys

import liftpose.lifter.utils as utils
import numpy as np
from torch.autograd import Variable
import torch
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(test_loader, model, criterion):
    """
    Compute test results using a given network (model)

    Parameters
    ----------
    test_loader : pytorch class
        DataLoader iterable class.
    model : pytorch neural network
        Trained network.
    criterion : torch class
        Loss function (L1 loss by default).

    Returns
    -------
    losses.avg : float
        Average test loss.
    all_output : np array
        Network prediction.
    all_target : np array
        Ground truth.
    all_input : np array
        Network input.
    all_bool : boolean np array
        Visible (high quality) key points.

    """
    losses = utils.AverageMeter()
    model.eval()

    all_output, all_target, all_input, all_bool = [], [], [], []

    for i, (inps, tars, good_keypts, keys) in enumerate(test_loader):

        # make prediction with model
        inputs = Variable(inps.to(device))
        targets = Variable(tars.to(device))
        outputs = model(inputs)

        # calculate loss
        outputs[~good_keypts] = 0
        targets[~good_keypts] = 0
        loss = criterion(targets, outputs)
        losses.update(loss.item(), inputs.size(0))
            
        all_output.append(outputs.data.cpu().numpy())
        all_input.append(inputs.data.cpu().numpy())
        all_target.append(targets.data.cpu().numpy())
        all_bool.append(good_keypts)

    all_input = np.vstack(all_input)
    all_output = np.vstack(all_output)
    all_bool = np.vstack(all_bool)
    all_target = np.vstack(all_target)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)

    return (
        losses.avg,
        all_output,
        all_target,
        all_input,
        all_bool,
    )