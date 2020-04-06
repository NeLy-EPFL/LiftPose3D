import numpy as np
import src.utils as utils
from tqdm import tqdm
from torch.autograd import Variable
from src.procrustes import get_transformation
from src.normalize import unNormalizeData, get_coords_in_dim


def test(test_loader, model, criterion, stat_z, procrustes=False):
    losses = utils.AverageMeter()

    model.eval()

    all_dist, all_output, all_target, all_input = [], [], [], []

    for i, (inps, tars) in enumerate(tqdm(test_loader)):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(non_blocking=True))

        #make prediction with model
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        # calculate accuracy
        targets_z = get_coords_in_dim(stat_z['target_sets'], 1)
        targets = unNormalizeData(targets.data.cpu().numpy(), stat_z['mean'], stat_z['std'], targets_z)
        outputs = unNormalizeData(outputs.data.cpu().numpy(), stat_z['mean'], stat_z['std'], targets_z)
        
        targets = targets[:,targets_z]
        outputs = outputs[:,targets_z]
        
#        if procrustes:
#            for ba in range(inps.size(0)):
#                gt = targets[ba].reshape(-1, 3)
#                out = outputs[ba].reshape(-1, 3)
#                _, Z, T, b, c = get_transformation(gt, out, True)
#                out = (b * out.dot(T)) + c
#                outputs[ba, :] = out.reshape(1, 51)

        sqerr = (outputs - targets) ** 2
        distance = np.sqrt(sqerr)
        
#        n_pts = int(len(stat_3d['dim_use']))
#        distance = np.zeros((sqerr.shape[0], n_pts)/3)
#        for k in np.arange(0, n_pts):
#            distance[:, k] = np.sqrt(np.sum(sqerr[:, 3*k:3*k + 3], axis=1))

        #group and stack
        all_dist.append(distance)
        all_output.append(outputs)
        all_target.append(targets)
        all_input.append(inputs.data.cpu().numpy())
    
    all_dist, all_output, all_target, all_input = \
    np.vstack(all_dist), np.vstack(all_output), np.vstack(all_target), np.vstack(all_input)
    
    #mean errors
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)

    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err, joint_err, all_output, all_target, all_input