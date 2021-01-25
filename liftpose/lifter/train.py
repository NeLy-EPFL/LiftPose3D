from tqdm import tqdm
import liftpose.lifter.utils as utils
from torch.autograd import Variable
import torch.nn as nn
import logging
import sys
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s:[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        
        '''
        for param_group in optimizer.param_groups:
            print(param_group[‘lr’])
        for param_group in :
            print(param_group[‘lr’])
        '''
        pbar.set_description(
            "Epoch {:03d} | LR {:8.5f} | Loss Test {:8.5f} | Loss Train {:8.5f}|".format(
                epoch, list(optimizer.param_groups)[0]['lr'], 0 if loss_test is None else loss_test, losses.avg
            )
        )
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        # make prediction with model
        inputs = Variable(inps.to(device))
        targets = Variable(tars.to(device))
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
