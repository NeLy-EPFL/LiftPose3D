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
import liftpose.lifter.log as log
from liftpose.lifter.log import save_ckpt
from liftpose.lifter.model import LinearModel, weight_init
from liftpose.lifter.data_loader_fun import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def network_main(opt, augmentation=None):
    """
    Main network training function.

    Parameters
    ----------
    opt : class
        Training options (see opt.py for default values).
    augmentation : list of functions, optional
        List of augmentation functions (see augmentation.py). The default is None.

    Returns
    -------
    None.

    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info(f"Training on the device: {device}")

    start_epoch = 0
    loss_best = 9999999999
    glob_step = 0
    lr_now = opt.lr

    # data loading
    stat_3d = torch.load(os.path.join(opt.data_dir, "stat_3d.pth.tar"))
    input_size = stat_3d["input_size"]
    output_size = stat_3d["output_size"]

    # save options
    log.save_options(opt, opt.out_dir)

    # create and initialise model
    model = LinearModel(
        input_size=input_size,
        output_size=output_size,
        p_dropout=opt.dropout,
        drop_inp=opt.drop_input,
    )
    model = model.to(device)
    model.apply(weight_init)
    criterion = nn.L1Loss(reduction="mean").to(device)
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
        loss_best = ckpt["err"]
        glob_step = ckpt["step"]
        lr_now = ckpt["lr"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("ckpt loaded (epoch: {} | err: {})".format(start_epoch, loss_best))

    log_file = "log_test.txt" if opt.test else "log_train.txt"

    if opt.resume:
        io = log.Logger(os.path.join(opt.out_dir, log_file), resume=True)
    else:
        io = log.Logger(os.path.join(opt.out_dir, log_file))
        io.set_names(["epoch", "lr", "loss_train", "loss_test"])

    # loader for testing
    test_loader = DataLoader(
        dataset=data_loader(
            data_path=opt.data_dir, is_train=False
        ),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True,
    )

    # test
    if opt.test:
        (
            loss_test,
            outputs,
            targets,
            inputs,
            good_keypts,
        ) = test(test_loader, model, criterion)

        logger.info(
            "Saving results: {}".format(
                os.path.join(opt.out_dir, "test_results.pth.tar")
            )
        )
        torch.save(
            {
                "loss": loss_test,
                "output": outputs,
                "target": targets,
                "input": inputs,
                "good_keypts": good_keypts,
            },
            open(os.path.join(opt.out_dir, "test_results.pth.tar"), "wb"),
        )
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
            loss_test, _, _, _, _ = test(test_loader, model, criterion)

            # update log file
            io.append(
                [epoch + 1, lr_now, loss_train, loss_test],
                ["int", "float", "float", "float"],
            )

            # save ckpt
            is_best = loss_test < loss_best
            loss_best = min(loss_test, loss_best)
            save_ckpt(
                {
                    "epoch": epoch + 1,
                    "lr": lr_now,
                    "step": glob_step,
                    "err": loss_best,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path=opt.out_dir,
                is_best=is_best,
            )

        io.close()