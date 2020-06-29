import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import src.utils as utils

def predict(test_loader, model, criterion, stat):

    model.eval()

    all_output, all_input = [], []

    for i, inps in enumerate(tqdm(test_loader)):
        inputs = Variable(inps.cuda())

        #make prediction with model
        outputs = model(inputs)
        
        # undo normalisation
        targets_1d = stat['targets_1d']
        outputs = utils.unNormalizeData(outputs.data.cpu().numpy(), stat['mean'][targets_1d], stat['std'][targets_1d])
        
        #group and stack
        all_output.append(outputs)
        all_input.append(inputs.data.cpu().numpy())
    
    all_output, all_input = np.vstack(all_output), np.vstack(all_input)

    return all_output, all_input