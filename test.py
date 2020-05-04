import numpy as np
import src.utils as utils
from tqdm import tqdm
from torch.autograd import Variable
from src.procrustes import get_transformation
from src.normalize import unNormalizeData, get_coords_in_dim


def test(test_loader, model, criterion, stat, procrustes=False):
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

        # undo normalisation to alculate accuracy in real units
        dimensions = get_coords_in_dim(stat['target_sets'], 3)
        targets = unNormalizeData(targets.data.cpu().numpy(), stat['mean'], stat['std'], dimensions)
        outputs = unNormalizeData(outputs.data.cpu().numpy(), stat['mean'], stat['std'], dimensions)
        
        targets = targets[:, dimensions]
        outputs = outputs[:, dimensions]
        
#        if procrustes:
#            for ba in range(inps.size(0)):
#                gt = targets[ba].reshape(-1, 3)
#                out = outputs[ba].reshape(-1, 3)
#                _, Z, T, b, c = get_transformation(gt, out, True)
#                out = (b * out.dot(T)) + c
#                outputs[ba, :] = out.reshape(1, 51)

        sqerr = (outputs - targets) ** 2
#        distance = np.sqrt(sqerr)
        
        n_pts = int(len(dimensions)/3)
        distance = np.zeros_like(sqerr)
        for k in range(n_pts):
            distance[:, k] = np.sqrt(np.sum(sqerr[:, 3*k:3*k + 3], axis=1))

        #group and stack
        all_dist.append(distance)
        all_output.append(outputs)
        all_target.append(targets)
        all_input.append(inputs.data.cpu().numpy())
    
    all_dist, all_output, all_target, all_input = \
    np.vstack(all_dist), np.vstack(all_output), np.vstack(all_target), np.vstack(all_input)
    
    #mean errors
    joint_err = np.mean(all_dist[all_dist>0], axis=0)
    ttl_err = np.mean(joint_err)

    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err, joint_err, all_output, all_target, all_input