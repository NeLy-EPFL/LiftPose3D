import os
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from liftpose.vision_3d import world_to_camera, project_to_camera
from liftpose.preprocess import anchor_to_root, remove_roots

class data_loader(Dataset):
    def __init__(self, data_path, is_train=True, noise=None, eangle=None, predict=False):
        """
        data_path: path to dataset
        is_train: load train/test dataset
        noise: std. of additive zero-mean Gaussian noise used in training
        predict: only predict but do not test
        """

        self.is_train = is_train
        self.noise = noise
        self.predict = predict
        self.eangle = eangle

        self.train_inp, self.test_inp, = [], []
        self.train_out, self.test_out = [], []
        self.train_keypts, self.test_keypts = [], []
        self.test_keys, self.train_keys = [], []
        

        if is_train: # load training data
            self.train_stat_2d = torch.load(os.path.join(data_path, "stat_2d.pth.tar"))
            self.train_3d, self.train_bool = torch.load(os.path.join(data_path, "train_3d.pth.tar"))
            
            if eangle is not None:
                self.train_stat_3d = torch.load(os.path.join(data_path, 'stat_3d.pth.tar'))
            else:
                self.train_2d = torch.load(os.path.join(data_path, 'train_2d.pth.tar'))
            
            for key in self.train_3d.keys():
                num_f = self.train_3d[key].shape[0]
                if eangle is None:
                    assert (
                        self.train_3d[key].shape[0] == self.train_2d[key].shape[0]
                    ), "(training) 3d & 2d shape not matched"
                for i in range(num_f):
                    if eangle is None:
                        self.train_inp.append(self.train_2d[key][i])
                    self.train_out.append(self.train_3d[key][i])

                    self.train_keypts.append(self.train_bool[key][i])
                    self.train_keys.append(key)

        else:  # load test data
            if not predict:
                self.test_3d, self.test_bool = torch.load(
                    os.path.join(data_path, "test_3d.pth.tar")
                )
            self.test_2d = torch.load(os.path.join(data_path, "test_2d.pth.tar"))
            for key in self.test_2d.keys():
                # print(self.test_2d[key].shape)
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
            if self.eangle is not None:
                
                # outputs = project_to_eangle(self.train_stat_2d, self.train_stat_3d, outputs)
                
                mean_2d = self.train_stat_2d['mean']
                std_2d = self.train_stat_2d['std']
                vis = self.train_stat_2d['vis']
                f = self.train_stat_2d['focal_length']
                s = self.train_stat_2d['frame_size']
                mean_3d = self.train_stat_3d['mean']
                std_3d = self.train_stat_3d['std']
                axsorder = self.train_stat_2d['axsorder']
                
                #obtain random rotation matrices
                if len(vis)==2:
                    lr = np.random.binomial(1,0.5)
                    if lr:
                        eangle = self.eangle[0]
                        vis = np.array(vis[0])
                    else:
                        eangle = self.eangle[1]
                        vis = np.array(vis[1])
                else:
                    vis = np.array(vis[0])
                    eangle = self.eangle[0]
                    
                a = np.random.uniform(low=eangle[0][0], high=eangle[0][1])
                b = np.random.uniform(low=eangle[1][0], high=eangle[1][1])
                c = np.random.uniform(low=eangle[2][0], high=eangle[2][1])
                R = Rot.from_euler(axsorder, [[a, b, c]], degrees=True).as_matrix()
    
                rcams = {'R': R, 
                         'tvec': np.array([0, 0, 117]),
                         'vis': vis}
                
                outputs = outputs[None,:]
                outputs = world_to_camera(outputs, rcams)
                inputs = project_to_camera(outputs, cam_par=(f[0], f[1], s[0], s[1]))
                
                # anchor points to body-coxa (to predict legjoints wrt body-coxas)
                inputs, _ = anchor_to_root( {'inputs': inputs}, self.train_stat_2d['roots'], self.train_stat_2d['target_sets'], 2)
                outputs, _ = anchor_to_root( {'outputs': outputs}, self.train_stat_2d['roots'], self.train_stat_2d['target_sets'], 3)
                
                inputs = inputs['inputs']
                outputs = outputs['outputs']
                
                # Standardize each dimension independently
                np.seterr(divide='ignore', invalid='ignore')
                inputs -= mean_2d
                inputs /= std_2d
                outputs -= mean_3d
                outputs /= std_3d
      
                #remove roots
                inputs, _ = remove_roots({'inputs': inputs}, self.train_stat_2d['target_sets'], 2)
                outputs, _ = remove_roots({'outputs': outputs}, self.train_stat_2d['target_sets'], 3)
                 
                inputs = inputs['inputs']
                outputs = outputs['outputs']
                
                inputs = torch.from_numpy(inputs[0,:]).float()
                outputs = torch.from_numpy(outputs[0,:]).float()
                
            else:
                inputs = torch.from_numpy(self.train_inp[index]).float()
            if self.noise is not None:
                std = self.train_stat_2d["std"][self.train_stat_2d["targets_2d"]]
                inputs += torch.from_numpy(
                    np.random.normal(0, self.noise / std, size=inputs.shape)
                ).float()
                
            good_keypts = torch.from_numpy(self.train_keypts[index])
            keys = self.train_keys[index]

        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            if self.predict:
                #print(inputs.shape)
                outputs = torch.from_numpy(np.zeros((36)))
                good_keypts = torch.from_numpy(np.zeros((36)).astype(bool))
                keys = torch.from_numpy(np.array(0))
            else:
                outputs = torch.from_numpy(self.test_out[index]).float()
                good_keypts = torch.from_numpy(self.test_keypts[index])
                keys = self.test_keys[index]

        return inputs, outputs, good_keypts, keys

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)

