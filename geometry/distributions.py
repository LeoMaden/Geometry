import numpy as np


def _calc_clustering(a, b, N, r, delta):
    if not ((r is not None) ^ (delta is not None)):
        raise Exception("Must provide one of `r` or `delta`")

    if r is not None:
        delta = (b - a) * ((1 - r) / (1 - r ** (N - 1)))
    elif delta is not None:
        A = (b - a) / delta
        coef = np.zeros(N)
        coef[0] = 1
        coef[-2] = -A
        coef[-1] = A - 1
        roots = np.roots(coef)
        r = np.max(roots[roots.imag == 0]).real

    spacing = np.geomspace(delta, delta * r ** (N - 2), N - 1)  # type: ignore
    return spacing, r, delta


def cluster_left(a: float, b: float, N: int, *, r=None, delta=None):
    spacing, r, delta = _calc_clustering(a, b, N, r, delta)
    arr = np.r_[a, a + np.cumsum(spacing)]
    arr[-1] = b
    return arr


def cluster_right(a: float, b: float, N: int, *, r=None, delta=None):
    spacing, r, delta = _calc_clustering(a, b, N, r, delta)
    spacing = spacing[::-1]
    arr = np.r_[a, a + np.cumsum(spacing)]
    arr[-1] = b
    return arr


def double_sided(a, b, N: int, alpha, *, r=None, delta=None):
    N_centre = int(alpha * N)
    N_left = (N - N_centre) // 2
    N_right = N_left
    N_centre = N - N_left - N_right

    if not ((r is not None) ^ (delta is not None)):
        raise Exception("Must provide one of `r` or `delta`")

    if r is not None:
        C = 2 * (1 - r ** (N_left - 1)) + (1 - r) * (N_centre + 1) * r ** (N_left - 2)
        delta = (1 - r) * (b - a) / C
    elif delta is not None:
        coef = np.zeros(N_left)
        coef[0] = delta * (N_centre + 3)
        coef[1] = -delta * (N_centre + 1)
        coef[-2] = -(b - a)
        coef[-1] = b - a - 2 * delta
        roots = np.roots(coef)
        r = max(roots[roots.imag == 0]).real

    geom_width = delta * ((1 - r ** (N_left - 1)) / (1 - r))
    b1 = a + geom_width
    a2 = b - geom_width

    left = cluster_left(a, b1, N_left, delta=delta)
    right = cluster_right(a2, b, N_right, delta=delta)
    centre = np.linspace(b1, a2, N_centre + 2)

    arr = np.r_[left, centre[1:-1], right]
    return arr
