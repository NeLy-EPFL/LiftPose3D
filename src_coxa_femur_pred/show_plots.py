import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, cycler

import data_utils

joints_name = ["Body-Coxa", "Coxa-Femur", "Femur-Tibia", "Tibia-Tarsus", "Tarsus tip"]

train_dir = "experiments_depth/model_50epochs_size1024_dropout0.5_0.001/"
with open(train_dir+"losses.pkl", 'rb') as f:
    losses = pickle.load(f)
with open(train_dir+"errors.pkl", 'rb') as f:
    errors = pickle.load(f)
with open(train_dir+"joint_errors.pkl", 'rb') as f:
    joint_errors = pickle.load(f)

plt.plot(losses)
plt.title("Loss function")
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Loss (mm)", fontsize=13)
plt.minorticks_on()
plt.grid(True, which='major', axis='y')
plt.grid(True, which='minor', axis='y', linestyle='--')
plt.savefig("plots/loss.png")
plt.show()

plt.plot(errors)
plt.title("Average joint error")
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Error (mm)", fontsize=13)
plt.minorticks_on()
plt.grid(True, which='major', axis='y')
plt.grid(True, which='minor', axis='y', linestyle='--')
plt.savefig("plots/error.png")
plt.show()

joints = [[] for i in range(data_utils.DF_NUM_JOINTS)]
for je in joint_errors:
    je = np.insert(je, 0, 0)
    for i in range(data_utils.DF_NUM_JOINTS):
        joints[i].append(je[i])
cmap = plt.cm.coolwarm
nj = data_utils.DF_JOINTS_PER_LEG
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0.7, 1, nj)))
for leg in range(6):
    for i in range(leg*nj, leg*nj+nj):
        plt.plot(joints[i])
    legend = []
    for i in range(nj):
        legend.append(joints_name[i])
  
    plt.title("Limb number %d"%(leg+1))
    plt.legend(legend, prop={'size': 9})
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("L2 error (mm)", fontsize=13)
    plt.minorticks_on()
    plt.grid(True, which='major', axis='y')
    plt.grid(True, which='minor', axis='y', linestyle='--')
    title = "plots/limb"+str(leg+1)+".png"
    plt.savefig(title)
    plt.show()
