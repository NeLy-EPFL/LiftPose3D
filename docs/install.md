# Installation
You can download the code for liftpose by cloning the Github reprository.

```git clone https://github.com/NeLy-EPFL/LiftPose3D```

Once downloaded, you can start the installation. First, change your directory into the LiftPose3D folder,

```cd LiftPose3D```

All the necessary dependencies for liftpose are given with environment.yml. The following command will create a new conda environment named liftpose. It will also install all the necessary packages.

```conda env create -f environment.yml```

Once installed, activate the environment, using

```conda activate liftpose```

Then, you can install the liftpose package using pip, first making sure you are in the root folder and then running

```pip install -e .```

Additionaly, you might need to setup your [GPU drivers](https://www.nvidia.com/Download/index.aspx), and install [the specific version of Pytorch](https://pytorch.org/get-started/previous-versions/), compatible with your driver versions in order to use your GPU.
You can check if your GPU setup is working properly by running  ```python -c "import torch; torch.cuda.is_available()"``` on your terminal. The terminal should print ```True``` in case of a successful installation.
