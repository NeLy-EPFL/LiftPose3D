import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

def predict(test_loader, model, criterion, stat):

    model.eval()

    all_output, all_input = [], []

    for i, inps in enumerate(tqdm(test_loader)):
        inputs = Variable(inps.cuda())

        #make prediction with model
        outputs = model(inputs)
        
        # undo normalisation
        dim=1
        dimensions = get_coords_in_dim(stat['target_sets'], dim)
        outputs = unNormalizeData(outputs.data.cpu().numpy(), stat['mean'], stat['std'], dimensions)
        outputs = outputs[:,dimensions]

        #group and stack
        all_output.append(outputs)
        all_input.append(inputs.data.cpu().numpy())
    
    all_output, all_input = np.vstack(all_output), np.vstack(all_input)

    return all_output, all_input


def get_coords_in_dim(targets, dim):
    
    if len(targets)>1:
      dim_to_use = []
      for i in targets:
          dim_to_use += i
    else:
      dim_to_use = targets
  
    dim_to_use = np.array(dim_to_use)
  
    return dim_to_use


def unNormalizeData(data, data_mean, data_std, dim_to_use):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing
  """

  data *= data_std[dim_to_use]
  data += data_mean[dim_to_use]
  
  T = data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality
  orig_data = np.zeros((T, D), dtype=np.float32)
  orig_data[:, dim_to_use] = data
  
  return orig_data