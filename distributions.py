import numpy as np


def _calc_clustering(a, b, N, r, delta):
    if not ((r is not None) ^ (delta is not None)):
        raise Exception("Must provide one of `r` or `delta`")

    if r is not None:
        delta = (b - a) / (r ** (N - 1) - 1)
    elif delta is not None:
        r = ((b - a + delta) / delta) ** (1 / (N - 1))

    spacing = np.geomspace(delta, b - a + delta, N)  # type: ignore
    return spacing, r, delta


def cluster_left(a: float, b: float, N: int, *, r=None, delta=None):
    spacing, r, delta = _calc_clustering(a, b, N, r, delta)
    print("left", f"{r=}")
    arr = a - delta + spacing  # type: ignore
    return arr


def cluster_right(a: float, b: float, N: int, *, r=None, delta=None):
    spacing, r, delta = _calc_clustering(a, b, N, r, delta)
    spacing = -spacing[::-1]
    arr = b - (-delta) + spacing  # type: ignore
    return arr


def double_sided(a, b, N: int, alpha, *, r=None, delta=None):
    N_centre = int(alpha * N)
    N_left = (N - N_centre) // 2
    N_right = N_left
    N_centre = N - N_left - N_right

    if not ((r is not None) ^ (delta is not None)):
        raise Exception("Must provide one of `r` or `delta`")

    if r is not None:
        C = (N_centre + 3) * r ** (N_left - 1) - (N_centre + 1) * r ** (N_left - 2) - 2
        delta = (b - a) / C
    elif delta is not None:
        coef = np.zeros(N_left)
        coef[0] = delta * (N_centre + 3)
        coef[1] = -delta * (N_centre + 1)
        coef[-1] = a - b - 2 * delta
        roots = np.roots(coef)
        r = roots[roots.imag == 0][0].real

    geom_width = (r ** (N_left - 1) - 1) * delta
    b1 = a + geom_width
    a2 = b - geom_width

    print(f"{b1=}, {a2=}")
    print(f"{a2-b1=}")

    Delta = (r ** (N_left - 1) - r ** (N_left - 2)) * delta
    print("Delta", Delta)
    print("b1 should be:", a + (r ** (N_left - 1) - 1) * delta)
    print("a2-b1 should be:", (N_centre + 1) * Delta)

    left = cluster_left(a, b1, N_left, delta=delta)
    right = cluster_right(a2, b, N_right, delta=delta)
    centre = np.linspace(b1, a2, N_centre + 2)

    arr = np.r_[left, centre[1:-1], right]
    return arr
