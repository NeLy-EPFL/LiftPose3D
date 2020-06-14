import numpy as np
import src.utils as utils
from tqdm import tqdm
from torch.autograd import Variable
from src.normalize import unNormalizeData, get_coords_in_dim


def test(test_loader, model, criterion, stat):
    losses = utils.AverageMeter()

    model.eval()

    all_mae, all_output, all_target, all_input, all_keys, all_bool = [], [], [], [], [], []

    for i, (inps, tars, vis_bool, keys) in enumerate(tqdm(test_loader)):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))

        #make prediction with model
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        
        outputs[~vis_bool] = 0
        targets[~vis_bool] = 0

        # undo normalisation to calculate accuracy in real units
        dim = 3
        dimensions = get_coords_in_dim(stat['target_sets'], dim)
        targets = unNormalizeData(targets.data.cpu().numpy(), stat['mean'], stat['std'], dimensions)
        outputs = unNormalizeData(outputs.data.cpu().numpy(), stat['mean'], stat['std'], dimensions)
        
        targets = targets[:, dimensions]
        outputs = outputs[:, dimensions]

        abs_err = np.abs(outputs - targets)

        n_pts = int(len(dimensions)/dim)
        mae = np.zeros_like(abs_err)
        for k in range(n_pts):
            mae[:, k] = np.mean(abs_err[:, dim*k:dim*(k + 1)], axis=1)

        #group and stack
        all_mae.append(mae)
        all_output.append(outputs)
        all_target.append(targets)
        all_input.append(inputs.data.cpu().numpy())
        all_bool.append(vis_bool)
        all_keys.append(keys)
    
    all_mae, all_output, all_target, all_input, all_bool, all_keys = \
    np.vstack(all_mae), np.vstack(all_output), np.vstack(all_target), np.vstack(all_input), np.vstack(all_bool), np.vstack(all_keys)
    
    #mean errors
    all_mae[all_mae == 0] = np.nan
    joint_err = np.nanmean(all_mae, axis=0)
    ttl_err = np.nanmean(joint_err)

    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err, joint_err, all_output, all_target, all_input, all_bool, all_keys