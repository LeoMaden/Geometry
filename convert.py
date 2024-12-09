import numpy as np


def xrrt_to_xyz(xrrt):
    x, r, rt = xrrt.T
    t = rt / r
    y = r * np.cos(t)
    z = r * np.sin(t)
    xyz = np.c_[x, y, z]
    return xyz
