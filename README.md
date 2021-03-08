# LiftPose3D
<p align="center">
  <img src="images/fig1ad.png" width="720">
</p>

The tool for transforming a single 2D pose to 3D coordinates on labaratory animals using a deep neural network.

For the theoretical background and for more details on the following examples have a look at our paper:
[LiftPose3D, a deep learning-based approach for transforming 2D to 3D pose in laboratory experiments](https://www.biorxiv.org/content/10.1101/2020.09.18.292680v1)

1. [Installation](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/install.md)
2. [Downloading the Datasets](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/dataset.md)
3. [Citing LiftPose3D](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/cite.md)

## Configuration file (param.yaml)
We define two short configuration files. First one is always named as param.yaml and placed in the example folder. It holds the information of root and target joints, together with information used in visualizing the animal. Notice that bone information, or the connected joints, are only used for visualization and not during training. You can safely remove vis subheading and still train and test with liftpose3d. 
  ```yaml
  data:
      roots: [0, 5, 10]
      target_sets: [
                  [1, 2, 3, 4],
                  [6, 7, 8, 9],
                  [11, 12, 13, 14],
            ]

  vis:
      colors: [
                  [186, 30, 49], 
                  [201, 86, 79], 
                  [213, 133, 121],
                  [15, 115, 153],
                  [26, 141, 175],
                  [117, 190, 203],
            ]
      bones: [
                  [0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [10, 11],
                  [11, 12],
                  [12, 13],
                  [13, 14],] 
  ```
  
  Second, we define the root folder for the data and the output folder are defined in a different python dictionary, inside the training script/notebook. This dictionary further defines the and used by load.py script which loads the necessary data for the. 
  
  ```python
  par_train = {'data_dir'       : '/data/LiftPose3D/fly_tether/data_DF3D/',
               'out_dir'        : './out/',
               'train_subjects' : [1],
               'test_subjects'  : [6],
               'actions'        : ['all'],
               'cam_id'         : [0,1,2,4,5,6]
               }
  ```
  
## Training
LiftPose3D accepts data in two different formats. 
1. A single numpy array in the shape of [N J 3], where N is number of poses and J is number of joints. 
2. A single dictionary, where the keys are strings and values are numpy arrays 

Under the hood, liftpose.main.train_np transforms it's input into a dictionary file and passes it to liftpose.main.train.

The following python scripts are valid uses of liftpose3d.

  ```python
  import liftpose.main.train_np as liftpose3d_train
  import numpy as np
  n_points, n_joints = 100, 5
  train_2d, test_2d = np.random.rand((n_points, n_joints, 2)), np.random.rand((n_points, n_joints, 2))
  train_3d, test_3d = np.random.rand((n_points, n_joints, 3)), np.random.rand((n_points, n_joints, 3))
  
  liftpose3d_train(train_2d, test_2d, train_3d, test_3d)
  ```
  
  This call train a deep neural network to predict the 3d pose, given 2d pose, and will save results in the 'out' folder, relative to the path where liftpose3d is called.
  
  ```python
  import liftpose.main.train as liftpose3d_train
  import numpy as np
  n_points, n_joints = 100, 5
  train_2d, test_2d = np.random.rand((n_points, n_joints, 2)), np.random.rand((n_points, n_joints, 2))
  train_3d, test_3d = np.random.rand((n_points, n_joints, 3)), np.random.rand((n_points, n_joints, 3))
  
  train_2d = {"some_string": train_2d}
  train_3d = {"some_string": train_3d}
  
  test_2d = {"some_other_string": test_2d}
  test_3d = {"some_other_string": test_3d}
  
  roots = [0]
  target_sets = [1,2,3,4]
  
  liftpose3d_train(train_2d, test_2d, train_3d, test_3d, roots, target_sets)
  ```
  This will result in the same training as with the previous example. Currently we support train_3d and test_3d to have 1 or 3 dimensions.
  During training, liftpose3d will log minimal information, such as IO information or start of the network training. Furthermore it will write intermediate     results into the output folder.
  
## Inspecting the training  
  
  Once the training is done, you can visualize the loss curves by reading the training logs and calling the function.
  
## Testing the network
  import liftpose.main.test as liftpose3d_test
  
## Visualizing the 3D pose

## More complicated use cases
  You can adjust all the necessary parameters of the training .

