# LiftPose3D, a deep learning-based approach for transforming 2D to 3D pose in laboratory experiments

<!--- ![video_5](https://user-images.githubusercontent.com/20509861/110424090-876c2180-80a2-11eb-87cc-38309236bf83.gif) --->
<p align="center">
  <img align="center" width="500" height="500" src="https://user-images.githubusercontent.com/20509861/110424218-bc787400-80a2-11eb-8164-61a5bf1085fe.gif">
</p>

A tool for transforming a single 2D pose to 3D coordinates on labaratory animals using a deep neural network. Current way to acquire 3D pose is by multi-view triangulation of deep network-based 2D pose estimates. This requires multiple, synchronised cameras and elaborate calibration protocols. LiftPose3D overcomes these barriers by reconstructing 3D poses from a single 2D camera view. For the theoretical background and for more details, please have a look at our [paper](https://www.biorxiv.org/content/10.1101/2020.09.18.292680v1).

## Starting-Up
1. [Installation](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/install.md)
2. [LiftPose3D Paper](https://www.biorxiv.org/content/10.1101/2020.09.18.292680v1)
3. [Downloading the Datasets](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/dataset.md)
4. [Citing LiftPose3D](https://github.com/NeLy-EPFL/LiftPose3D/blob/package_sem/docs/cite.md)

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
  
  You can have a closer look at the ```train``` function, [default values and much longer documentation here](https://github.com/NeLy-EPFL/LiftPose3D/blob/7548b391e80bebb10e5ae6dce8624022a4019f53/liftpose/main.py#L97).
  

## Configuration file (param.yaml)
We define two short configuration files. First one is always named as param.yaml and placed in the example folder. It holds the information of root and target joints, together with information used in visualizing the animal. Notice that bone information, or the connected joints, are only used for visualization and not during training. You can safely remove vis subheading and still train and test with liftpose3d. 
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
               'cam_id'         : [0,1,2,4,5,6]
               }
  ```
  

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
  data = torch.load(os.path.join(par['out_dir'], "test_results.pth.tar"))
  stat_2d, stat_3d = (
    torch.load(os.path.join(par['out_dir'], "stat_2d.pth.tar")),
    torch.load(os.path.join(par['out_dir'], "stat_3d.pth.tar")),
  )
  test_3d_gt, test_3d_pred, _ = load_test_results(data, stat_2d, stat_3d)
  ```
  This will return two numpy arrays, ```test_3d_gt``` and ```test_3d_pred```. test_3d_gt is the same array as test_3d, whereas ```test_3d_pred``` has the predictions from the LiftPose3D. You can test the error in your predictions simply by doing 
  ```python
  np.mean(np.linalg.norm(test_3d_gt - test_3d_pred ,2))
  ```
## Visualizing the 3D pose
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
 This should output something similar to :
 
## More complicated use cases
1. Augmentations
2. Training arguments, opts
  You can adjust all the necessary parameters of the training .
3. good_keypts

from liftpose.plot import plot_pose_3d
