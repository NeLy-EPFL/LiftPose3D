#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset

class data_loader(Dataset):
    def __init__(self, data_path):
        """
        data_path: path to dataset
        """
        self.data_path = data_path
        self.test_inp = []
        self.train_meta, self.test_meta = [], []

        # loading data             
        self.test_2d = torch.load(os.path.join(data_path, 'test_2d.pth.tar'))
        for k2d in self.test_2d.keys():
            (sub, act, fname) = k2d
            print(fname)
            k2d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k2d
            num_f, _ = self.test_2d[k2d].shape
            for i in range(num_f):
                self.test_inp.append(self.test_2d[k2d][i])    
        

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.test_inp[index]).float()
        return inputs
    

    def __len__(self):
        return len(self.test_inp)       