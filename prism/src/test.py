import numpy as np
import src.utils as utils
from tqdm import tqdm
from torch.autograd import Variable
from src.normalize import unNormalizeData, get_coords_in_dim


def test(test_loader, model, criterion, stat):
    losses = utils.AverageMeter()

    model.eval()

    all_dist, all_output, all_target, all_input, all_keys, all_bool = [], [], [], [], [], []

    for i, (inps, tars, bool_LR, keys) in enumerate(tqdm(test_loader)):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))

        #make prediction with model
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        
        #evaluate left or right side limbs based on fly orientation
        out = outputs.data.cpu().numpy().copy()
        tar = targets.data.cpu().numpy().copy()
        
        outputs[bool_LR] = 0
        targets[bool_LR] = 0
        
        # undo normalisation to calculate accuracy in real units
        dim = 1
        dimensions = get_coords_in_dim(stat['target_sets'], dim)
        targets = unNormalizeData(targets.data.cpu().numpy(), stat['mean'], stat['std'], dimensions)
        outputs = unNormalizeData(outputs.data.cpu().numpy(), stat['mean'], stat['std'], dimensions)
        
        targets = targets[:, dimensions]
        outputs = outputs[:, dimensions]

        sqerr = (outputs - targets) ** 2

        n_pts = int(len(dimensions)/dim)
        distance = np.zeros_like(sqerr)
        for k in range(n_pts):
            distance[:, k] = np.sqrt(np.sum(sqerr[:, dim*k:dim*(k + 1)], axis=1))

        #group and stack
        all_dist.append(distance)
        all_output.append(out)
        all_target.append(tar)
        all_input.append(inputs.data.cpu().numpy())
        all_bool.append(bool_LR)
        all_keys.append(keys)
    
    all_dist, all_output, all_target, all_input, all_bool, all_keys = \
    np.vstack(all_dist), np.vstack(all_output), np.vstack(all_target), np.vstack(all_input), np.vstack(all_bool), np.vstack(all_keys)
    
    #mean errors
    all_dist[all_dist == 0] = np.nan
    joint_err = np.nanmean(all_dist, axis=0)
    ttl_err = np.nanmean(joint_err)

    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err, joint_err, all_output, all_target, all_input, all_bool, all_keys