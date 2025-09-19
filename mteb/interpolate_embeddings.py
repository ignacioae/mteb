from scipy.spatial import geometric_slerp
import numpy as np

def slerp(u: np.array, v: np.array, t):
    """
    Perform spherical linear interpolation (slerp) between two vectors.

    Parameters:
    u (np.array): The starting vector.
    v (np.array): The ending vector.
    t (float or np.array): The interpolation parameter, can be a single float or an array of floats.

    Returns:
    np.array: The interpolated vector(s).
    """
    u_norm = u.astype(np.float64)/np.linalg.norm(u.astype(np.float64))
    v_norm = v.astype(np.float64)/np.linalg.norm(v.astype(np.float64))

    return geometric_slerp(u_norm, v_norm, t)