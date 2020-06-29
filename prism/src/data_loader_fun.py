import os
import torch
from torch.utils.data import Dataset
import numpy as np

class data_loader(Dataset):
    def __init__(self, data_path, is_train=True, noise=None):
        """
        data_path: path to dataset
        is_train: load train/test dataset
        """
        self.data_path = data_path
        self.is_train = is_train
        self.noise = noise

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_LR, self.test_LR = [], []
        self.test_keys, self.train_keys = [], []

        if self.is_train: # load training data
            self.train_3d, self.train_bool_LR = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
            self.train_2d = torch.load(os.path.join(data_path, 'train_2d.pth.tar'))
            self.train_stat = torch.load(os.path.join(data_path, 'stat_2d.pth.tar'))
            for key in self.train_2d.keys():
                num_f, num_d = self.train_3d[key].shape
                assert self.train_3d[key].shape[0] == self.train_2d[key].shape[0], '(training) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.train_inp.append(self.train_2d[key][i])
                    self.train_out.append(self.train_3d[key][i])
                    self.train_keys.append(key)
                    mask = np.ones(num_d, dtype=bool)
                    if self.train_bool_LR[key][i]:
                        mask[:num_d//2] = 0
                    else:
                        mask[num_d//2:] = 0
                        
                    self.train_LR.append(mask)  
                    
        else:# load test data
            self.test_3d, self.test_bool_LR = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
            self.test_2d = torch.load(os.path.join(data_path, 'test_2d.pth.tar'))
            for key in self.test_2d.keys():
                num_f, num_d = self.test_3d[key].shape
                assert self.test_2d[key].shape[0] == self.test_3d[key].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.test_inp.append(self.test_2d[key][i])
                    self.test_out.append(self.test_3d[key][i])
                    self.test_keys.append(key)
                    mask = np.ones(num_d, dtype=bool)
                    if self.test_bool_LR[key][i]:
                        mask[:num_d//2] = 0
                    else:
                        mask[num_d//2:] = 0
                        
                    self.test_LR.append(mask)
        
        
    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            if self.noise is not None:
                std = self.train_stat['std'][self.train_stat['targets_2d']]
                inputs += torch.from_numpy(np.random.normal(0, self.noise/std, size=inputs.shape)).float()
            outputs = torch.from_numpy(self.train_out[index]).float()
            bool_LR = torch.from_numpy(self.train_LR[index])
            keys = self.train_keys[index]
            
        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()
            bool_LR = torch.from_numpy(self.test_LR[index])
            keys = self.test_keys[index]

        return inputs, outputs, bool_LR, keys


    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)       