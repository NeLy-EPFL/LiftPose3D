# Installation
All the necessary dependencies for liftpose are given with environment.yml. The following command will create a new conda environment named liftpose. It will also install all the necessary packages.
```conda env create -f environment.yml```
Once installed, activate the environment, using
```conda activate liftpose```.
Then, you can install the liftpose package using pip.
```pip install -e .```
Additionaly, you might need to setup your GPU drivers, and install the specific drivers of pytorch. Check [https://www.nvidia.com/Download/index.aspx] and [https://pytorch.org/get-started/previous-versions/] for more details. 
You can check if your GPU setup is working properly by running  ```python -m "import torch; torch.cuda.is_available()"``` 

