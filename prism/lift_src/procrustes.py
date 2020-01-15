import pickle
import data_utils
import numpy as np


def apply_transformation(pts, R=None, t=None, s=None, tform=None):
    if tform is not None:
        R = tform["rotation"]
        s = tform["scale"]
        t = tform["translation"]
    return s * np.dot(pts, R) + t

def middle_point(xyz1, xyz2):
    new_xyz = np.zeros(xyz1.shape)
    for coord in range(3):
        new_xyz[:,coord] = (xyz1[:,coord] + xyz2[:,coord]) / 2
    return new_xyz

def procrustes_separate(pts, template, reflection='best', verbose=False, scaling=True,
                        return_transf=False):
    '''
    Performs procrustes seperately for each side
    '''
    m_left = np.arange(0, 15)
    points3d_gt_left = template[:, m_left].copy()
    points3d_pred_left = pts[:, m_left].copy()
    pts_t_left = procrustes(pts=points3d_pred_left,
                            body_coxa_idx=body_coxa_idx,
                            template=points3d_gt_left,
                            verbose=verbose,
                            scaling=scaling,
                            reflection=reflection,
                            return_transf=False)

    m_right = np.arange(15, 30)
    points3d_gt_right = template[:, m_right].copy()
    points3d_pred_right = pts[:, m_right].copy()
    pts_t_right, tform = procrustes(pts=points3d_pred_right,
                                    body_coxa_idx=body_coxa_idx,
                                    template=points3d_gt_right,
                                    verbose=verbose,
                                    scaling=scaling,
                                    reflection=reflection,
                                    return_transf=True)

    pts3d_proc = np.zeros_like(pts)
    pts3d_proc[:, m_left] = pts_t_left.copy()
    pts3d_proc[:, m_right] = pts_t_right.copy()

    return pts3d_proc

def procrustes_get_tform(pts, template, reflection='best', verbose=False, scaling=True,
                        return_transf=False):
    '''
    Performs procrustes seperately for each side
    '''
    pts, new_body_coxa = virtual_coxa_femur(pts)
    template, _ = virtual_coxa_femur(template)
    body_coxa_idx = []
    for bc in new_body_coxa[:3]:
        body_coxa_idx.append(bc)
        body_coxa_idx.append(bc+1)
    
    m_left = np.arange(0, 15)
    points3d_gt_left = template[:, m_left].copy()
    points3d_pred_left = pts[:, m_left].copy()
    pts_t_left, tform_left = procrustes(pts=points3d_pred_left,
                                   body_coxa_idx=body_coxa_idx,
                                   template=points3d_gt_left,
                                   verbose=verbose,
                                   scaling=scaling,
                                   reflection=reflection,
                                   return_transf=True)

    m_right = np.arange(15, 30)
    points3d_gt_right = template[:, m_right].copy()
    points3d_pred_right = pts[:, m_right].copy()
    pts_t_right, tform_right = procrustes(pts=points3d_pred_right,
                                    body_coxa_idx=body_coxa_idx,
                                    template=points3d_gt_right,
                                    verbose=verbose,
                                    scaling=scaling,
                                    reflection=reflection,
                                    return_transf=True)

    pts3d_proc = np.zeros_like(pts)
    pts3d_proc[:, m_left] = pts_t_left.copy()
    pts3d_proc[:, m_right] = pts_t_right.copy()

    return m_left, tform_left, m_right, tform_right


def procrustes(pts, body_coxa_idx, template=None, reflection='best',
        verbose=False, scaling=True, return_transf=False):
    
    if template is None:
        template = read_template_pose3d()

    template_bc = template[:, body_coxa_idx]
    pts_bc = pts[:, body_coxa_idx]

    template_bc = np.median(template_bc, axis=0)
    pts_bc = np.median(pts_bc, axis=0)

    d, Z, tform = procrustes_(template_bc, pts_bc, reflection=reflection, scaling=scaling)
    R_b, s_b, t_b = tform["rotation"], tform["scale"], tform["translation"]

    pts_t = apply_transformation(pts.copy(), R_b, t_b, s_b)

    if verbose:
        print("Body-coxa index:", body_coxa_idx)
        print("Tform,", tform)

    if return_transf:
        return pts_t, tform
    else:
        return pts_t


def procrustes_(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform
