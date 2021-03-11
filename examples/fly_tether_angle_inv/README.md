## Angle invariant lifter network for tethered fly

You can use this script to train a network that can lift 3D poses for tethered flies from any camera angle. As long as the focal length is long enough or the camera is far enough you can use any camera. We used 94mm focal length cameras.

Uses the same data as fly_tether.

- Download the DeepFly3D dataset "fly_tether.zip" from the drive link: https://drive.google.com/drive/u/1/folders/1KDgkZgfJOsRc4bQ0P0wGbpwbQ_RcHpd5
- Unzip the file. In angle_invariant_lifter.ipynb set the "data_dir" value inside the config dictionary as the location of the unzipped DeepFly3D dataset folder.
- Using sample_cameras.ipynb, visualise the allowed camera angles with respect to the position of the fly in world coordinates (uses pose.pkl)
- Using angle_invariant_lifter.ipynb, you can train, test and visualize the results. You can change the parameters like amount of data used, the Euler angle ranges.

# Reproducing the Figure
- To create the figure, run errors.ipynb notebook, first by changing the "data_dir" value inside the first cell.