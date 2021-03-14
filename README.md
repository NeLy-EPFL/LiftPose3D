# LiftPose3D, a deep learning-based approach for transforming 2D to 3D pose in laboratory experiments

<!--- ![video_5](https://user-images.githubusercontent.com/20509861/110424090-876c2180-80a2-11eb-87cc-38309236bf83.gif) --->
<p align="center">
  <img align="center" width="500" height="500" src="https://user-images.githubusercontent.com/20509861/110424218-bc787400-80a2-11eb-8164-61a5bf1085fe.gif">
</p>

LiftPose3D is a tool for transforming a 2D poses to 3D coordinates on labaratory animals. Classical approaches based on triangulation require synchronised acquisition from multiple cameras and elaborate calibration protocols. By contrast, LiftPose3D can reconstruct 3D poses from 2D poses from a single camera, in some instances without having to know the camera position or the type of lens used. For the theoretical background and details, have a look at our [paper](https://www.biorxiv.org/content/10.1101/2020.09.18.292680v1).

## Starting-Up
1. [Installation](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/install.md)
2. [LiftPose3D Paper](https://www.biorxiv.org/content/10.1101/2020.09.18.292680v1)
3. [Downloading the Datasets](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/dataset.md)
4. [Citing LiftPose3D](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/cite.md)

## Requirements
To train LiftPose3D, most ideally you need (A) a 3D pose library, (B) corresponding 2D poses from the camera that you will use for lifting and (C) camera matrices (extrinsic and intrinsic). 

If you do not have access to 
(A), you can use one of the provided datasets for your animal system,
(B), you can obtain 2D images via projection using your camera matrices (you will need to calibrate to obtain these)
(C), you can place your camera further away to assume weak perspective.

## Training
LiftPose3D accepts data in two different formats. An [N J 2] numpy array as input and [N J 3] numpy array as output, where N is number of poses and J is number of joints. If you have multiple experiments, you can provide dictionaries where the keys are strings and values are numpy arrays. You will also need at least one root joint and a set of target sets for each root joint. The network will predict the joints in the target sets relative to the root joitns.

You can train a network with the following generic syntax.

  ```python
  import liftpose.main.train
  import numpy.random.rand
  n_points, n_joints = 100, 5
  train_2d, test_2d = rand((n_points, n_joints, 2)), rand((n_points, n_joints, 2))
  train_3d, test_3d = rand((n_points, n_joints, 3)), rand((n_points, n_joints, 3))
  
  train_2d = {"experiment 1": train_2d}
  train_3d = {"experiment 1": train_3d}
  
  test_2d = {"experiment 2": test_2d}
  test_3d = {"experiment 2": test_3d}
  
  roots = [0]
  target_sets = [1,2,3,4]
  
  train(train_2d, test_2d, train_3d, test_3d, roots, target_sets)
  ```

The following outputs will be saved in a folder ```out``` relative to the path where liftpose3d is called.
  
  You can have a closer look at the ```train``` function, [default values and much longer documentation here](https://github.com/NeLy-EPFL/LiftPose3D/blob/7548b391e80bebb10e5ae6dce8624022a4019f53/liftpose/main.py#L97).
  
## Configuration (param.yaml)
You can customize training by passing an extra argument ```training_kwargs``` in ```train()```.

  ```python
  training_kwargs={"epochs":15, #train for 15 epochs
                   "resume":True, #resume training where it was stopped
                   "load":par['out_dir'] + '/ckpt_last.pth.tar'}, #load last training checkpoint
  ```

Several training augmentation options can also be specified in the argument ```augmentation```. Using the following option, the training will ignore the input ```train_2d``` and insted generate 2D poses by projecting the 3D poses to ordered Euler angles within the range ```'eangles'```. 

  ```python
  from liftpose.lifter.augmentation import random_project
  
  angle_aug = {'eangles': {0: [[-10,10], [-10, 10], [-10,10]]}, #range of Euler angles (this is a dictionary indexed by an integer which specifies the camera identify)
               'axsorder': 'zyx', #order of rotations
               'vis': None, #used in case not all joints are visible from a given camera
               'tvec': None, #vecto from camera centered to world coordinate frames
               'intr': None} # camera intrinsic matrix
  
  aug = [random_project(**angle_aug)]
  ```

Have a look at other options for augmentation in ```liftpose.lifter.augmentation``` and see one of the examples for various implementations.

## Inspecting the training  
  
  Once the training is done, you can visualize the loss curves by reading the training logs and calling the function. The training information is saved under the ```train_log.txt```, which can easily be read using a csv reader. Alternatively, we already provide functions to read and visualize the file.
  
  ```python
  from liftpose.plot import read_log_train, plot_log_train
  epoch, lr, loss_train, loss_test, err_test = read_log_train(par['out_dir'])
  plot_log_train(plt.gca(), loss_train, loss_test, epoch)
  ```
  This will plot the training and test losses during the training.
  <p align="center">
   <img src="https://user-images.githubusercontent.com/20509861/110373519-dfc60380-804f-11eb-9bbe-6db6f17c5fc6.png" width="360">
  </p>

  
## Testing the network
  You can test the network.
  ```python
  import liftpose.main.test as liftpose3d_test
  liftpose3d_test(par['out_dir'])
  ```
  This will run the network with the test data provided during the ```liftpose3d_train``` call. It will save the results inside the ```test_results.pkl``` file. 
  In case you want to test the network in new data, you can run
  
  ```python
  import liftpose.main.test as liftpose3d_test
  liftpose3d_test(par['out_dir'], test_2d, test_3d)
  ```
  where you provide the ```test_2d``` and ```test_3d``` numpy arrays. This will overwrite the previous ```test_results.pkl``` file, if there is any.
  
  We also provide a simple interface for loading the test results from the ```test_results.pkl``` file. 
  
  ```python
  from liftpose.postprocess import load_test_results
  test_3d_gt, test_3d_pred, _ = load_test_results(par['out_dir'])
  ```
  This will return two numpy arrays, ```test_3d_gt``` and ```test_3d_pred```. test_3d_gt is the same array as test_3d, whereas ```test_3d_pred``` has the predictions from the LiftPose3D. You can test the error in your predictions simply by doing 
  ```python
  np.mean(np.linalg.norm(test_3d_gt - test_3d_pred ,2))
  ```
  
  you can also use the violin plot code provided with liftpose3d to plot the error distribution: 
  
  
 ```python
  from liftpose.plot import violin_plot
  violin_plot(plt.gca(), test_3d_gt, test_3d_pred, test_keypoints=np.ones_like(test_3d_pred_ord), joints_name=par_data["vis"]["names"])
  ```
  
  <p align="center">
  <img align="center" width="300" height="300" src="https://user-images.githubusercontent.com/20509861/110426701-d61bba80-80a6-11eb-885c-b73012c17fd3.png">
  </p>


## Visualizing the 3D pose

Visualization is specified in the file param.yaml which is placed in the example folder. It holds the information of root and target joints, together with information used in visualizing the animal. Notice that bone information, or the connected joints, are only used for visualization and not during training, you can have a closer look at ```plot_pose_3d``` function to see how the bone and color parameters are used. You can safely remove vis subheading and still train and test with liftpose3d. 
  ```yaml
  data:
      roots: [0]
      target_sets: [[1, 2, 3, 4]]

  vis:
      colors: [[186, 30, 49]]
      bones: [[0, 1], [1, 2], [2, 3], [3, 4]]
      limb_id: [0, 0, 0, 0, 0]
  ```
  
  Second, we define the root folder for the data and the output folder are defined in a different python dictionary, inside the training script/notebook. This dictionary further defines the and used by load.py script which loads the necessary data for the. 
  
  ```python
  par_train = {'data_dir'       : '/data/LiftPose3D/fly_tether/data_DF3D/',
               'out_dir'        : './out/',
               'train_subjects' : [1],
               'test_subjects'  : [6],
               'actions'        : ['all'],
               'cam_id'         : [0,1,2,4,5,6]}
  ```
  
Once you read the network predictions on the test data, you can visualize the target and prediction test data using the provided 3d visualization function, which uses the bone and color information you provided in the configuration file to draw the animal:
 ```python
fig = plt.figure(figsize=plt.figaspect(1), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-75, azim=-90)

t = 0
plot_pose_3d(ax=ax, tar=test_3d_gt[t], 
            pred=test_3d_pred[t], 
            bones=par_data["vis"]["bones"], 
            limb_id=par_data["vis"]["limb_id"], 
            colors=par_data["vis"]["colors"])
 ```
 This should output something similar to:
 TODO: replace with a better pictuere
 
   <p align="center">
  <img align="center" width="300" height="300" src="https://user-images.githubusercontent.com/20509861/110427610-5131a080-80a8-11eb-81bc-b11867ee0e9f.png">
  </p>

## TODO More complicated use cases
### Augmentations
### Training arguments, opts
  You can adjust all the necessary parameters of the training .

### Training with subset of points
In case you want to remove remove some 2d/3d points from used in the training, you can pass ```train_keypts``` argument into the ```train``` function, which has the same shape as ```train_3d```. Alternatively, in case you have missing keypoints, you can convert them to ```np.NaN```. 
