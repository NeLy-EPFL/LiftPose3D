- Download the DeepFly3D dataset "fly_tether.zip" from the drive link: https://drive.google.com/drive/u/1/folders/1KDgkZgfJOsRc4bQ0P0wGbpwbQ_RcHpd5
- Unzip the file. Set the "data_dir" value inside the config dictionary as the location of the unzipped DeepFly3D dataset folder.
- Using fly_tether.ipynb, you can train, test and visualize the results. You can change the parameters like amount of data used, the camera used to train and the animals used in the training by changing the corresponding parameters in the config dictionary.

## Reproducing the Figure
- First, edit run fly_tether_train.py config dictionary to set the data path. This should create three output folders, out_cam[1], out_cam[2] and out_cam[5].
- Download retriangulation.pkl from the drive link and place it under the folder examples/fly_tether
- Then, run fly_tether_train.py. This will create 3 different networks for 3 different camers.
- To create the figure, run Figure_1EG.ipynb notebook, first by changing the "data_dir" value inside the first cell.

## Reproducing the Video
- First, edit run fly_tether_train.py config dictionary to set the data path. This should create three output folders, out_cam[1], out_cam[2] and out_cam[5].
- Then, run fly_tether_train.py. This will create 3 different networks for 3 different cameras.
- First, change the data_dir inside the notebook fly_tether_make_video.ipynb and then run the cells inside the notebook.