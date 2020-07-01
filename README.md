# LiftFly3D

Tool for transforming 2D keypoints from a single viewpoint to 3D coordinates using deep neural networks.

For the theoretical background and for more details on the following examples have a look at our paper:

paper citation/hyperlink here

Don't forget to cite us if you find our work useful.

The best way to use our code is to adapt one of our examples below to your application.

## Data format

Currently LiftFly3D is set up to work with data supplied as a Python dictionary and saved as a pickle file. Conveniently, this is same format used for DeepFly3D, our pipeline used for multi-camera triangulation. The dictionary must contain two keys (1) 'points3d': a numpy array of dimension AxBx3 containing the 3D coordinates in a global reference frame, where A is the number of frames, B is the number of keypoints and 3 are the x, y, z coordinates, and (2) 'points2d': a numpy array of dimension CxAxBx2 containing the 2D coordinates in camera centric reference frame, where C is the number of cameras and A, B as before.

Refer to sample_data.pkl for an example.

## Examples

To reproduce our results in the following examples, the provided Python scripts must be run in the numbered order. 

### 3D pose of a tethered fly on a spherical-treadmill using two cameras

The relevant code is under the folder ```/DF3D``` and the corresponding data with the pre-trained network can be downloaded here.

#### Scripts: 

```1_LiftFly3D_preprocess.py``` - this file preprocesses the training and test data

You must specify the following

- TRAIN_SUBJECTS = [0,1,2,3,4,5] #flies used for training
- TEST_SUBJECTS  = [6,7] #flies used for testing
- cam_id = 1 #camera with respect to which we train the network
- data_dir = '/directory_of_your_data'
- out_dir = '/directory_to_save_output'
- actions = ['PR', 'MDN_CsCh', 'aDN_CsCh']
- camera_matrices = '/folder_of_camera_parameters/cameras.pkl'
- interval = None #timeframes to be used (None for all data or np.arange(200,700) for frames between 200 and 700)
- dims_to_consider = [i for i in range(38) if i not in [15,16,17,18,34,35,36,37]] #joint coordinates to consider (note we exclude points for some body parts 
- target_sets = [[ 1,  2,  3,  4],  [6,  7,  8,  9], [11, 12, 13, 14],
               [16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]] #group points to be lifted with respect to the same coordinate system
- origins = [0, 5, 10, 15, 20, 25] #joint representing the origin or each coordinate system

```2_LiftFly3D_main.py``` - this file trains the network or predicts test data

At minimum you need

To train
```
python 2_LiftFly3D_main.py --data_dir /directory_to_save_output --out /directory_to_save_output
```
To test
```
python 2_LiftFly3D_main.py --data_dir /directory_to_save_output --out /directory_to_save_output --test --load /directory_to_save_output/ckpt_best.pth.tar
```
Refer to ```/src/opt.py``` for more options.



### 3D pose of a freely behaving fly in a prism-mirror setup using partial triangulated data



### Transfer learning-based 3D pose in a low resolution ventral camera setup
