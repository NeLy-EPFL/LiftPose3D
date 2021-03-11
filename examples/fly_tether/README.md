## Tethered fly
- Download the DeepFly3D dataset "fly_tether.zip" from the drive link: https://drive.google.com/drive/u/1/folders/1KDgkZgfJOsRc4bQ0P0wGbpwbQ_RcHpd5
- Unzip the file. In fly_tether.ipynb set the "data_dir" value inside the config dictionary as the location of the unzipped DeepFly3D dataset folder.
- Using fly_tether.ipynb, you can train, test and visualize the results. You can change the parameters like amount of data used, the camera used to train and the animals used in the training.

# Reproducing the Figure
- Download retriangulation.pkl from the drive link and place it in the 'out_folder'
- To create the figure, run errors.ipynb notebook, first by changing the "data_dir" value inside the first cell.