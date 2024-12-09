from typing import Callable
import numpy as np

from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar


def calc_tangents(curve: NDArray):
    """Calculate normalised tangent vectors along a curve in 2D

    Parameters
    ----------
    curve : ndarray, shape (N, 2)
        Coordinates of points along the curve.

    Returns
    -------
    ndarray, shape (N, 2)
        Tangent vectors at each point
    """
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]  # Normalize length
    return tangents


def calc_normals(curve: NDArray):
    """Calculate unit normal vectors along a curve in 2D

    Parameters
    ----------
    curve : ndarray, shape (N, 2)
        Coordinates of points along the curve.

    Returns
    -------
    ndarray, shape (N, 2)
        Normal vectors at each point
    """
    tangents = calc_tangents(curve)
    rot_90_ccw = np.array([[0, -1], [1, 0]])
    normals = tangents @ rot_90_ccw.T
    # normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize length
    return normals


def segment_lengths(curve: NDArray):
    """Calculate length of each segment of the curve

    The length is calculated as the sum of the lengths of each line segment
    making up the curve.

    Parameters
    ----------
    curve : ndarray, shape (N, D)
        Coordinates of points along the curve.

    Returns
    -------
    ndarray, shape (N-1,)
        The length of each segment along the curve.
    """
    return np.linalg.norm(np.diff(curve, axis=0), axis=1)


def total_length(curve: NDArray) -> float:
    """Calculate length of a curve in `D` dimensions.

    The length is calculated as the sum of the lengths of each line segment
    making up the curve.

    Parameters
    ----------
    curve : NDArray, shape (N, D)
        Coordinates of points along the curve.

    Returns
    -------
    float
        The length of the curve.
    """
    deltas = segment_lengths(curve)
    return np.sum(deltas)


def cumulative_length(curve: NDArray, normalise: bool = False) -> NDArray:
    """Calculate cumulative length of each point along a curve in `D` dimensions.

    The length is calculated as the sum of the lengths of each line segment
    making up the curve.

    Parameters
    ----------
    curve : ndarray, shape (N, D)
        Coordinates of points along the curve.
    normalise : bool, default = False
        Whether to normalise the length of the curve to 1

    Returns
    -------
    ndarray, shape (N,)
        The cumulative length of each point along the curve.
    """
    deltas = segment_lengths(curve)
    cumulative = np.r_[0, np.cumsum(deltas)]
    if normalise == True:
        cumulative /= cumulative[-1]
    return cumulative


def curve_interp_spline(
    curve, deg=1, normalise: bool = False
) -> Callable[[NDArray], NDArray]:
    """Create a spline fit of the curve where the abscissa is the distance
    along the curve.

    Parameters
    ----------
    curve : ndarray, shape (N, D)
        Coordinates of points along the curve.
    normalise : bool, default = False
        Whether to normalise the length of the curve to 1

    Returns
    -------
    ndarray, shape (M, D)
        The coordinates of the interpolated points.
    """
    cum_len = cumulative_length(curve, normalise=normalise)
    return make_interp_spline(cum_len, curve, k=deg)


def curve_interp(curve, x, deg=1, normalise: bool = False) -> NDArray:
    """Interpolate a curve at the lengths along the curve given by `x`.

    Parameters
    ----------
    curve : ndarray, shape (N, D)
        Coordinates of points along the curve.
    normalise : bool, default = False
        Whether to normalise the length of the curve to 1

    x : ndarray, shape (M,)
        The normalised lengths along the curve; 0 is the start of the curve and 1
        is the end.

    Returns
    -------
    ndarray, shape (M, D)
        The coordinates of the interpolated points.
    """
    return curve_interp_spline(curve, deg, normalise)(x)
