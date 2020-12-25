from tqdm import tqdm
import liftpose.lifter.utils as utils
from torch.autograd import Variable
import torch.nn as nn
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s:[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
)


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    loss_test,
    lr_init=None,
    lr_now=None,
    glob_step=None,
    lr_decay=None,
    gamma=None,
    max_norm=True,
):

    losses = utils.AverageMeter()
    model.train()
    pbar = tqdm(train_loader)
    for i, (inps, tars, good_keypts, keys) in enumerate(pbar):
        pbar.set_description(
            "Epoch {} | Loss Test {:.5g} | Loss Train {:.5g}|".format(
                epoch, 0 if loss_test is None else loss_test, losses.avg
            )
        )
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        # make prediction with model
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))
        outputs = model(inputs)

        # evaluate high confidence keypoints only
        outputs[~good_keypts] = 0
        targets[~good_keypts] = 0

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    return glob_step, lr_now, losses.avg
