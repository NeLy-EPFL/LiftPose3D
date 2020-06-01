#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset
import numpy as np

class data_loader(Dataset):
    def __init__(self, data_path, use_hg=True, is_train=True):
        """
        data_path: path to dataset
        use_hg: use stacked hourglass detections
        is_train: load train/test dataset
        """
        self.data_path = data_path

        self.is_train = is_train
        self.use_hg = use_hg

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_LR, self.test_LR = [], []
        self.train_meta, self.test_meta = [], []

        # loading data
        if self.use_hg:
            train_2d_file = 'train_2d_ft.pth.tar'
            test_2d_file = 'test_2d_ft.pth.tar'
        else:
            train_2d_file = 'train_2d.pth.tar'
            test_2d_file = 'test_2d.pth.tar'

        if self.is_train:
            # load train data
            self.train_3d, self.train_bool_LR = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
            self.train_2d = torch.load(os.path.join(data_path, train_2d_file))
            for k2d in self.train_2d.keys():
                (sub, act, fname) = k2d
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                num_f, num_d = self.train_3d[k3d].shape
                assert self.train_3d[k3d].shape[0] == self.train_2d[k2d].shape[0], '(training) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.train_inp.append(self.train_2d[k2d][i])
                    self.train_out.append(self.train_3d[k3d][i])
                    mask = np.ones(num_d, dtype=bool)
                    if self.train_bool_LR[k3d][i]:
                        mask[:int(num_d/2)] = 0
                    else:
                        mask[int(num_d/2):] = 0
                        
                    self.train_LR.append(mask)
                    
                    
        else:
            # load test data
            self.test_3d, self.test_bool_LR = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
            self.test_2d = torch.load(os.path.join(data_path, test_2d_file))
            for k2d in self.test_2d.keys():
                (sub, act, fname) = k2d
                print(fname)
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                num_f, num_d = self.test_3d[k3d].shape
                assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f):
                    
                    self.test_inp.append(self.test_2d[k2d][i])
                    self.test_out.append(self.test_3d[k3d][i])
                    mask = np.ones(num_d, dtype=bool)
                    if self.test_bool_LR[k3d][i]:
                        mask[:int(num_d/2)] = 0
                    else:
                        mask[int(num_d/2):] = 0
                        
                    self.test_LR.append(mask)
        
        
    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()
            bool_LR = torch.from_numpy(self.train_LR[index])
            
        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()
            bool_LR = torch.from_numpy(self.test_LR[index])

        return inputs, outputs, bool_LR


    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)       