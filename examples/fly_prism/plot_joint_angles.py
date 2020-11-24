import numpy as np
import matplotlib.pyplot as plt

idx_LF_body_coxa = 0
idx_LF_coxa_femur = 1
idx_LF_femur_tibia = 2
idx_LF_tibia_tarsus = 3
idx_LF_tarsal_claw = 4
idx_LM_body_coxa = 5
idx_LM_coxa_femur = 6
idx_LM_femur_tibia = 7
idx_LM_tibia_tarsus = 8
idx_LM_tarsal_claw = 9
idx_LH_body_coxa = 10
idx_LH_coxa_femur = 11
idx_LH_femur_tibia = 12
idx_LH_tibia_tarsus = 13
idx_LH_tarsal_claw = 14
idx_RF_body_coxa = 15
idx_RF_coxa_femur = 16
idx_RF_femur_tibia = 17
idx_RF_tibia_tarsus = 18
idx_RF_tarsal_claw = 19
idx_RM_body_coxa = 20
idx_RM_coxa_femur = 21
idx_RM_femur_tibia = 22
idx_RM_tibia_tarsus = 23
idx_RM_tarsal_claw = 24
idx_RH_body_coxa = 25
idx_RH_coxa_femur = 26
idx_RH_femur_tibia = 27
idx_RH_tibia_tarsus = 28
idx_RH_tarsal_claw = 29
idx_L_antenna = 30
idx_R_antenna = 31
idx_L_eye = 32
idx_R_eye = 33
idx_L_haltere = 34
idx_R_haltere = 35
idx_L_wing = 36
idx_R_wing = 37
idx_proboscis = 38
idx_neck = 39
idx_genitalia = 40

import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return (vector.T / np.linalg.norm(vector, axis=1)).T

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':: in degrees

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.degrees(np.arccos(np.clip((v1_u * v2_u).sum(axis=1), -1.0, 1.0)))

def main():
    points3d = np.load('/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/VV/points3d.npy')
    mean_eye = (points3d[:, idx_L_eye, :] + points3d[:, idx_R_eye, :])/2
    v1 = points3d[:, idx_neck, :] - mean_eye
    v2 = points3d[:, idx_L_antenna, :] - mean_eye
    angle_Lantenna_meanEye_neck = angle_between(v1, v2)

    import matplotlib.pyplot as plt
    plt.plot(angle_Lantenna_meanEye_neck)  # Plot list. x-values assumed to be [0, 1, 2, 3]
    plt.grid()
    plt.show()  # Optional wh

    # import matplotlib
    # import matplotlib.pyplot as plt
    # # Data for plotting
    # t = np.arange(0.0, 2.0, 0.01)
    # s = 1 + np.sin(2 * np.pi * t)
    # fig, ax = plt.subplots()
    # ax.plot(t, s)
    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #        title='About as simple as it gets, folks')
    # ax.grid()
    # fig.savefig("test.png")
    # plt.show()

    print('end')

if __name__ == "__main__":
    main()
