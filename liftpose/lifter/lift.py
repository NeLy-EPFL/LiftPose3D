import os
import sys
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import logging

from liftpose.lifter.test import test
from liftpose.lifter.train import train

# from liftpose.lifter.opt import Options
import liftpose.lifter.log as log
from liftpose.lifter.log import save_ckpt
from liftpose.lifter.model import LinearModel, weight_init
from liftpose.lifter.data_loader_fun import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def network_main(opt, augmentation=None):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info(f"Training on the device: {device}")

    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # data loading
    stat_3d = torch.load(os.path.join(opt.data_dir, "stat_3d.pth.tar"))
    input_size = stat_3d["input_size"]
    output_size = stat_3d["output_size"]

    # save options
    log.save_options(opt, opt.out_dir)

    # create and initialise model
    model = LinearModel(input_size=input_size, output_size=output_size)
    model = model.to(device)
    model.apply(weight_init)
    criterion = nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    logger.info(
        "total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1000000.0
        )
    )

    # load pretrained ckpt
    if opt.load:
        logger.info("loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt["epoch"]
        err_best = ckpt["err"]
        glob_step = ckpt["step"]
        lr_now = ckpt["lr"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    log_file = "log_test.txt" if opt.test else "log_train.txt"

    if opt.resume:
        io = log.Logger(os.path.join(opt.out_dir, log_file), resume=True)
    else:
        io = log.Logger(os.path.join(opt.out_dir, log_file))
        io.set_names(["epoch", "lr", "loss_train", "loss_test", "err_test"])

    # loader for testing and prediction
    test_loader = DataLoader(
        dataset=data_loader(
            data_path=opt.data_dir, is_train=False, predict=opt.predict,
        ),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True,
    )

    # test
    if opt.test | opt.predict:
        (
            loss_test,
            err_test,
            joint_err,
            all_err,
            outputs,
            targets,
            inputs,
            good_keypts,
        ) = test(test_loader, model, criterion, stat_3d, predict=opt.predict)

        logger.info(
            "Saving results: {}".format(
                os.path.join(opt.out_dir, "test_results.pth.tar")
            )
        )
        torch.save(
            {
                "loss": loss_test,
                "all_err": all_err,
                "test_err": err_test,
                "joint_err": joint_err,
                "output": outputs,
                "target": targets,
                "input": inputs,
                "good_keypts": good_keypts,
            },
            open(os.path.join(opt.out_dir, "test_results.pth.tar"), "wb"),
        )

        # if not opt.predict:
        #    logger.info("{:.4f}".format(err_test))

    else:
        # loader for training
        train_loader = DataLoader(
            dataset=data_loader(
                data_path=opt.data_dir, is_train=True, augmentation=augmentation,
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.job,
            pin_memory=True,
        )

        # loop through epochs
        cudnn.benchmark = True
        loss_test = None
        for epoch in range(start_epoch, opt.epochs):

            # train
            glob_step, lr_now, loss_train = train(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch=epoch,
                loss_test=loss_test,
                lr_init=opt.lr,
                lr_now=lr_now,
                glob_step=glob_step,
                lr_decay=opt.lr_decay,
                gamma=opt.lr_gamma,
                max_norm=opt.max_norm,
            )

            # test
            loss_test, err_test, _, _, _, _, _, _ = test(
                test_loader, model, criterion, stat_3d
            )

            # update log file
            io.append(
                [epoch + 1, lr_now, loss_train, loss_test, err_test],
                ["int", "float", "float", "float", "float"],
            )

            # save ckpt
            is_best = err_test < err_best
            err_best = min(err_test, err_best)
            save_ckpt(
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

        io.close()
