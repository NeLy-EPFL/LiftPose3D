## CAPTURE rat dataset
- Download the part of the CAPTURE dataset used in our paper https://drive.google.com/drive/folders/1qi8_c1YnlOzh7eWYXAG369iLtAS4iu1H?usp=sharing
- make sure you cite this dataset if you use it (see below)
- Note that image folders (and the corresponding zip files) named rat7M_*.zip are quite large (around 40GB). In case you don't want to wait, you can download only calibration and data folders. In case you want to reproduce the videos given in the LiftPose3D paper, you only need to download rat7M_e0.zip.
- Unzip the file. In capture.ipynb set the "data_dir" to point to the path of your dataset.
- Using capture.ipynb, you can train, test and visualize the results and compute the errors. You can change the parameters like amount of data used, the camera used to train and the animals used in the training.

citation for the CAPTURE dataset

@article{Marshall,
author = {Marshall, Jesse D and Aldarondo, Diego E and Dunn, Timothy W and Wang, William L and Berman, Gordon J and \"{O}lveczky, Bence P},
title = {Continuous Whole-Body 3{D} Kinematic Recordings across the Rodent Behavioral Repertoire},
journal = {Neuron},
year = {2021},
volume = {109},
number = {3},
pages = {420--437.e8},
month = feb
}
