import numpy as np
def plot_cameras_err(ax, err, mi, ma, cameras, curr_camera):
    # normalize the erros
    color = {k: (e - mi) / (ma - mi) for (k, e) in err.items()}
    color = {k: min(max(e,0),1) for (k, e) in color.items()}
    
    # draw each camera
    for c in cameras.keys():
        C = cameras[c]['C']
        if c != curr_camera:
            col = (color[c], 0, 1-color[c]) if c in color else (0,0,0)
        else:
            col = (0,1,0)
        s = ax.scatter(C[0], C[1], C[2], color=col, alpha=1 if c in color else 0.1)
        
def get_err(pt, pt_pred):
    return np.abs(pt - pt_pred).mean()

def err_for_frame(Data, cameras, frame_id, test_3d_gt, test_3d_pred):
    err = dict()
    for idx, (_, f, cam_id) in enumerate(Data.keys()):
        if f == frame_id:
            err[cam_id] = get_err(test_3d_gt[idx], test_3d_pred[idx])
            
    return err