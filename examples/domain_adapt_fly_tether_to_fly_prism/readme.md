## Lifting poses of freely behaving flies using a LiftPose3D network trained on tethered fly poses and domain adaptation

To begin with, use ```train_network.ipynb``` to train a LiftPose3D network with tethered fly training data. The process is exactly the same outlined in the ```fly_tether``` example.

Then, move to ```domain_adaptation.ipynb``` apply this network on freely behaving fly data. After both tethered fly (target domain) and freely behaving fly (source domain) datasets have been loaded, find nearest neighbors of poses of target domain poses in the source domain. The poses together with their nearest neighbors are used to find linear transformations. The number of poses used during fitting the linear transformations should be optimised such that, at test time, target-domain poses are mapped close to their nearest neighbors in the source domain. Map all poses to the source domain and lift using the pre-trained network.
