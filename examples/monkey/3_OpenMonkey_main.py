#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import sys

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from src.test import test
from src.train import train
from src.opt import Options
import src.log as log
from src.model import LinearModel, weight_init, SemGCN
from src.data_loader_fun import data_loader
from src.graph_utils import *


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.out_dir)

    # create and initialise model
    # parents = [1, 2, 7, 7, 5, 7, 5, -1, 8, 7, 7, 10, 7]
    # assert len(parents) == 13
    # adj = adj_mx_from_skeleton(13, parents)

    model = LinearModel(
        input_size=26,
        output_size=39,
        linear_size=opt.linear_size,
        num_stage=opt.num_stage,
        p_dropout=opt.dropout,
    )
    # groups = [[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]
    # model = SemGCN(adj, 128, num_layers=4, p_dropout=0.0, nodes_group=None)

    # model = SemGCN()
    model = model.cuda()
    model.apply(weight_init)
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    print(
        ">>> total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1000000.0
        )
    )

    # load pretrained ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt["epoch"]
        err_best = ckpt["err"]
        glob_step = ckpt["step"]
        lr_now = ckpt["lr"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    if opt.test:
        log_file = "log_test.txt"
    else:
        log_file = "log_train.txt"
    if opt.resume:
        logger = log.Logger(os.path.join(opt.out_dir, log_file), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.out_dir, log_file))
        logger.set_names(["epoch", "lr", "loss_train", "loss_test", "err_test"])

    # data loading
    print("\n>>> loading data")
    stat_3d = torch.load(os.path.join(opt.data_dir, "stat_3d.pth.tar"))

    # test
    if opt.test:
        test_loader = DataLoader(
            dataset=data_loader(data_path=opt.data_dir, is_train=False),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True,
        )

        loss_test, err_test, joint_err, all_err, outputs, targets, inputs = test(
            test_loader, model, criterion, stat_3d
        )

        print(os.path.join(opt.out_dir, "test_results.pth.tar"))
        torch.save(
            {
                "loss": loss_test,
                "all_err": all_err,
                "test_err": err_test,
                "joint_err": joint_err,
                "output": outputs,
                "target": targets,
                "input": inputs,
            },
            open(os.path.join(opt.out_dir, "test_results.pth.tar"), "wb"),
        )

        # print("train {:.4f}".format(err_train), end="\t")
        print("test {:.4f}".format(err_test), end="\t")
        sys.exit()

    # load datasets for training
    test_loader = DataLoader(
        dataset=data_loader(data_path=opt.data_dir, is_train=False),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True,
    )

    train_loader = DataLoader(
        dataset=data_loader(data_path=opt.data_dir, is_train=True, noise=opt.noise),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True,
        drop_last=False,
    )

    # loop through epochs
    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print("==========================")
        print(">>> epoch: {} | lr: {:.10f}".format(epoch + 1, lr_now))

        # train
        glob_step, lr_now, loss_train = train(
            train_loader,
            model,
            criterion,
            optimizer,
            lr_init=opt.lr,
            lr_now=lr_now,
            glob_step=glob_step,
            lr_decay=opt.lr_decay,
            gamma=opt.lr_gamma,
            max_norm=opt.max_norm,
        )

        loss_test, err_test, _, _, _, _, _ = test(
            train_loader, model, criterion, stat_3d
        )

        # test
        loss_test, err_test, _, _, _, _, _ = test(
            test_loader, model, criterion, stat_3d
        )

        # update log file
        logger.append(
            [epoch + 1, lr_now, loss_train, loss_test, err_test],
            ["int", "float", "float", "float", "float"],
        )

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        log.save_ckpt(
            {
                "epoch": epoch + 1,
                "lr": lr_now,
                "step": glob_step,
                "err": err_best,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path=opt.out_dir,
            is_best=is_best,
        )

    logger.close()


if __name__ == "__main__":
    import numpy as np

    # np.random.seed(0)
    import torch

    # torch.manual_seed(0)
    option = Options().parse()
    main(option)
