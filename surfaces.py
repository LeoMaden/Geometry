import numpy as np


# Normal to structured surface
def calc_normals(surf):
    i_tangent = np.gradient(surf, axis=0)
    i_tangent /= np.linalg.norm(i_tangent, axis=-1)[..., np.newaxis]

    j_tangent = np.gradient(surf, axis=1)
    j_tangent /= np.linalg.norm(j_tangent, axis=-1)[..., np.newaxis]

    normal = np.cross(i_tangent, j_tangent)
    return normal
