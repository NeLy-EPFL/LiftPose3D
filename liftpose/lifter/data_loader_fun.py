import os
import torch
from torch.utils.data import Dataset
import numpy as np


class data_loader(Dataset):
    def __init__(self, data_path, is_train=True, predict=False, augmentation=None):
        """
        data_path: path to dataset
        is_train: load train/test dataset
        noise: std. of additive zero-mean Gaussian noise used in training
        predict: only predict but do not test
        """

        self.is_train = is_train
        self.predict = predict
        self.augmentation = augmentation

        self.train_inp, self.test_inp, = [], []
        self.train_out, self.test_out, self.train_out_raw, self.test_out_raw = (
            [],
            [],
            [],
            [],
        )
        self.train_keypts, self.test_keypts = [], []
        self.test_keys, self.train_keys = [], []

        if is_train:  # load training data
            self.train_stat_2d = torch.load(os.path.join(data_path, "stat_2d.pth.tar"))
            self.train_stat_3d = torch.load(os.path.join(data_path, "stat_3d.pth.tar"))
            self.train_3d, self.train_bool, self.train_3d_raw = torch.load(
                os.path.join(data_path, "train_3d.pth.tar")
            )
            self.train_2d, self.train_2d_raw = torch.load(
                os.path.join(data_path, "train_2d.pth.tar")
            )

            for key in self.train_3d.keys():
                num_f = self.train_3d[key].shape[0]
                for i in range(num_f):
                    self.train_inp.append(self.train_2d[key][i])
                    self.train_out.append(self.train_3d[key][i])
                    self.train_out_raw.append(self.train_3d_raw[key][i])
                    self.train_keypts.append(self.train_bool[key][i])
                    self.train_keys.append(key)

        else:  # load test data
            if not predict:
                self.test_3d, self.test_bool, self.test_3d_raw = torch.load(
                    os.path.join(data_path, "test_3d.pth.tar")
                )
            self.test_2d, self.test_2d_raw = torch.load(
                os.path.join(data_path, "test_2d.pth.tar")
            )
            for key in self.test_2d.keys():
                num_f = self.test_2d[key].shape[0]
                for i in range(num_f):
                    self.test_inp.append(self.test_2d[key][i])
                    if not predict:
                        self.test_out.append(self.test_3d[key][i])
                        self.test_keypts.append(self.test_bool[key][i])
                        self.test_keys.append(key)

    def __getitem__(self, index):
        if self.is_train:
            outputs = torch.from_numpy(self.train_out[index]).float()
            outputs_raw = torch.from_numpy(self.train_out_raw[index]).float()
            inputs = torch.from_numpy(self.train_inp[index]).float()
        
            if self.augmentation is not None:
                for aug in self.augmentation:
                    inputs, outputs = aug(
                        inputs, outputs, outputs_raw, **self.get_aug_args()
                    )

            good_keypts = torch.from_numpy(self.train_keypts[index])
            keys = self.train_keys[index]
        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            if self.predict:
                outputs = torch.from_numpy(np.zeros((36)))
                good_keypts = torch.from_numpy(np.zeros((36)).astype(bool))
                keys = torch.from_numpy(np.array(0))
            else:
                outputs = torch.from_numpy(self.test_out[index]).float()
                good_keypts = torch.from_numpy(self.test_keypts[index])
                keys = self.test_keys[index]

        return inputs, outputs, good_keypts, keys

    def get_aug_args(self):
        mean_2d = self.train_stat_2d["mean"]
        std_2d = self.train_stat_2d["std"]
        mean_3d = self.train_stat_3d["mean"]
        std_3d = self.train_stat_3d["std"]
        roots = self.train_stat_2d.get("roots")
        target_sets = self.train_stat_2d.get("target_sets")
        targets_2d = self.train_stat_2d["targets_2d"]
        
        return {
            "stats": {
                "mean_2d": mean_2d,
                "std_2d": std_2d,
                "mean_3d": mean_3d,
                "std_3d": std_3d,
            },
            "roots": roots,
            "target_sets": target_sets,
            "targets_2d": targets_2d,
        }

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)

