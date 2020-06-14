#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset


class data_loader(Dataset):
    def __init__(self, data_path, is_train=True):
        """
        data_path: path to dataset
        use_hg: use stacked hourglass detections
        is_train: load train/test dataset
        """
        self.data_path = data_path
        self.is_train = is_train
        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.vis_train, self.vis_test = [], []
        self.test_keys, self.train_keys = [], []

        # loading data
        if self.is_train: #training data
            self.train_3d, vis_train = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
            self.train_2d = torch.load(os.path.join(data_path, 'train_2d.pth.tar'))
            for key in self.train_2d.keys():
                num_f, num_d = self.train_3d[key].shape
                assert self.train_3d[key].shape[0] == self.train_2d[key].shape[0], '(training) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.train_inp.append(self.train_2d[key][i])
                    self.train_out.append(self.train_3d[key][i])
                    self.train_keys.append(key)
                    self.vis_train.append(vis_train[key])
                    
        else: #test data
            self.test_3d, vis_test = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
            self.test_2d = torch.load(os.path.join(data_path, 'test_2d.pth.tar'))
            for key in self.test_2d.keys():
                num_f, num_d = self.test_3d[key].shape
                assert self.test_2d[key].shape[0] == self.test_3d[key].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.test_inp.append(self.test_2d[key][i])
                    self.test_out.append(self.test_3d[key][i])
                    self.test_keys.append(key)
                    self.vis_test.append(vis_test[key])
        
        
    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()
            bool_vis = torch.from_numpy(self.vis_train[index])
            keys = self.train_keys[index]
        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()
            bool_vis = torch.from_numpy(self.vis_test[index])
            keys = self.test_keys[index]

        return inputs, outputs, bool_vis, keys


    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)       