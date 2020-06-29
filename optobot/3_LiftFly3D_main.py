#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from src.predict import predict
from src.opt import Options
from src.model import LinearModel, weight_init
from src.data_loader_fun import data_loader


def main(opt):

    # create and initialise model
    model = LinearModel(input_size=36, output_size=18)
    model = model.cuda()
    model.apply(weight_init)
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    # load pretrained ckpt
    print(">>> loading ckpt from '{}'".format(opt.load))
    ckpt = torch.load(opt.load)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    
    # data loading
    print("\n>>> loading data")
    stat2d = torch.load(os.path.join(opt.data_dir, 'stat_3d.pth.tar'))
    
    # predict
    test_loader = DataLoader(
            dataset=data_loader(data_path=opt.data_dir),
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
            
    outputs, inputs = predict(test_loader, model, criterion, stat2d)
            
    torch.save({'output': outputs,
                'input': inputs}, 
                open(os.path.join(opt.out_dir,"test_results.pth.tar"), "wb"))
            
if __name__ == "__main__":
    option = Options().parse()
    main(option)