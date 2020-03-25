#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset

class data_loader(Dataset):
    def __init__(self, actions, data_path, use_hg=True, is_train=True):
        """
        actions: list of actions to use
        data_path: path to dataset
        use_hg: use stacked hourglass detections
        is_train: load train/test dataset
        """

        self.actions = actions
        self.data_path = data_path

        self.is_train = is_train
        self.use_hg = use_hg

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_meta, self.test_meta = [], []

        # loading data
        if self.use_hg:
            train_2d_file = 'train_2d_ft.pth.tar'
            test_2d_file = 'test_2d_ft.pth.tar'
            
        else:
            train_2d_file = 'train_2d.pth.tar'
            test_2d_file = 'test_2d.pth.tar'

        if self.is_train:
            data_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
            data_2d = torch.load(os.path.join(data_path, train_2d_file))
            self.train_3d = data_3d
            self.train_2d = data_2d           
        else:
            data_3d = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
            data_2d = torch.load(os.path.join(data_path, test_2d_file))
            self.test_3d = data_3d
            self.test_2d = data_2d
            
        for k2d in data_2d.keys():
            (sub, act, fname) = k2d
            k3d = k2d
            k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
            num_f, _ = data_2d[k2d].shape
            assert data_3d[k3d].shape[0] == data_2d[k2d].shape[0], '3d & 2d shape not matched'
            
            for i in range(num_f):
                if self.is_train:
                    self.train_inp.append(data_2d[k2d][i])
                    self.train_out.append(data_3d[k3d][i])
                    
                else:
                    self.test_inp.append(data_2d[k2d][i])
                    self.test_out.append(data_3d[k3d][i])


    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()

        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()

        return inputs, outputs

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)