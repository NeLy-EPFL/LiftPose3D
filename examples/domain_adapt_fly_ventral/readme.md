## Lift Drosophila poses in various ventral-view setups using a pretrained network and domain adaptation.

First, download the data from https://drive.google.com/drive/folders/1n0ipYJ8380LPsFr3-f8OzEZYob4IpuHK?usp=sharing and https://drive.google.com/drive/folders/1XsBXpGv2WL6Qs4gTDEmWDn7BLx0iBhMV?usp=sharing.

Use ```train_network.ipybn``` to train a LiftPose3D network using prism-mirror based Drosophila data. We use this data, rather than tethered fly data because this way only input poses need to be mapped to the network's domain. Since the lifted poses are already in te desired domain avoids unnecessary loss in accuracy.

Then use ```LiftPose3D_station.ipynb``` to load datasets, find nearest neighbors of poses in the target domain in the source (network) domain. Map 2D poses into the network's domain before lifting. 

The process is currently demonstrated for the LiftPose3D station, while other use cases ```FlyLimbTracker.ipynb```, ```fly_ventral_lowres.ipynb``` are lifted using precomputed networks.
