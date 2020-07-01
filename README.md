# LiftFly3D

Tool for transforming 2D keypoints from a single viewpoint to 3D coordinates using deep neural networks.

For the theoretical background and for more details on the following examples have a look at our paper:

paper citation/hyperlink here

Don't forget to cite us if you find our work useful.

Data format

Currently LiftFly3D is set up to work with data supplied as a python dictionary and saved as a pickle file. Conveniently, this is same format used for DeepFly3D, our pipeline used for multi-camera triangulation. The dictionary must contain two keys (1) 'points3d': a numpy array of dimension AxBx3 containing the 3D coordinates in a global reference frame, where A is the number of frames, B is the number of keypoints and 3 are the x, y, z coordinates, and (2) 'points2d': a numpy array of dimension CxAxBx2 containing the 2D coordinates in camera centric reference frame, where C is the number of cameras and A, B as before.

Refer to sample_data.pkl for an example.


You will need to provide your data in the same dictionary format as DeepFly3D.
