from intersect import intersection
from dataclasses import dataclass, field
from typing import Generic, Iterator, Self, TypeVar
import warnings
from scipy.integrate import cumulative_simpson, simpson
from scipy.interpolate import make_interp_spline, BSpline
from functools import cached_property, partial
from numpy.typing import ArrayLike, NDArray
import numpy as np


ROT_90_DEG = np.array([[0, -1], [1, 0]])


norm = partial(np.linalg.norm, axis=-1)
CLOSE_TOL = 1e-5


@dataclass(frozen=True)
class Curve:
    coords: NDArray  # Must be (N, D)
    param: NDArray  # Must be (N,)

    @classmethod
    def new(cls, coords, param) -> Self:
        return cls(coords, param)

    @classmethod
    def new_unit_speed(cls, coords) -> Self:
        delta = _curve_norm(np.diff(coords, axis=0))
        s = np.r_[0, np.cumsum(delta)]
        curve = cls(coords, s)
        if not curve.is_unit:
            curve._warn_not_unit()
        return cls(coords, s)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            msg = "Coords of curve must have shape (N, D)"
            raise ValueError(msg)
        if self.param.ndim != 1:
            msg = "Param of curve must have shape (N,)"
            raise ValueError(msg)
        if self.coords.shape[0] != self.param.shape[0]:
            msg = "Coords and param must have the same length"
            raise ValueError(msg)

    @cached_property
    def is_unit(self) -> bool:
        """True if this curve has unit speed along its length."""
        dot = _curve_deriv(self.coords, self.param)
        speed = _curve_norm(dot)
        return np.allclose(speed, 1.0, atol=CLOSE_TOL)

    def dot(self) -> Self:
        """Return a new curve which is the derivative of this curve with
        respect to the parameter.
        """
        return self.new(_curve_deriv(self.coords, self.param), self.param)

    def ddot(self) -> Self:
        """Return a new curve which is the second derivative of this curve with
        respect to the parameter.
        """
        return self.dot().dot()

    def tangent(self) -> "Self":
        """Return a new curve which is the unit tangent vector of this curve."""
        dot = self.dot()
        if self.is_unit:
            return dot
        speed = dot.norm()[:, np.newaxis]
        return self.new(dot.coords / speed, self.param)

    def curvature(self) -> NDArray:
        """Return the curvature at each point along the curve."""
        if self.is_unit:
            return _curve_norm(self.ddot().coords)
        dot = self.dot()
        ddot = dot.dot()
        A1 = ddot.coords * np.vecdot(dot.coords, dot.coords)[:, np.newaxis]
        A2 = dot.coords * np.vecdot(dot.coords, ddot.coords)[:, np.newaxis]
        A = _curve_norm(A1 - A2)
        B = dot.norm() ** 4
        return A / B

    def arc_length(self) -> NDArray:
        """Return the arc-length of each point along the curve, starting at zero."""
        if self.is_unit:
            return self.param
        dot = self.dot()
        delta = _curve_norm(np.diff(self.coords, axis=0))
        return np.r_[0, np.cumsum(delta)]

    def interpolate(self, t: ArrayLike, *args, **kwargs) -> Self:
        """Return a new curve formed by interpolating coordinates of this curve
        at the parameter values `t`.

        Additional arguments are passed to the Scipy function `make_interp_spline`.
        """
        t = np.asarray(t)
        spl: BSpline = make_interp_spline(self.param, self.coords, *args, **kwargs)
        new_coords = spl(t)
        return self.new(new_coords, t)

    def interpolate_equal(self) -> Self:
        """Return a new curve with normalised constant-speed parameterisation and
        equal arc-length between points.
        """
        t_equal = np.linspace(0, 1, len(self.param))
        return self.reparameterise_unit().normalise().interpolate(t_equal)

    def reparameterise(self, t: ArrayLike) -> Self:
        """Return a new curve with the parameter changed to `t`."""
        t = np.asarray(t)
        return self.new(self.coords, t)

    def norm(self) -> NDArray:
        """Return the vector norm of each vector in `coords`."""
        return _curve_norm(self.coords)

    def reparameterise_unit(self) -> Self:
        s = self.arc_length()
        new_curve = self.reparameterise(s)
        if not new_curve.is_unit:
            new_curve._warn_not_unit()
        return new_curve

    def normalise(self) -> Self:
        """Return a new curve which has parameter between 0 and 1."""
        p0, pN = self.param[[0, -1]]
        new_param = (self.param - p0) / (pN - p0)
        return self.reparameterise(new_param)

    def start(self) -> NDArray:
        return self.coords[0]

    def end(self) -> NDArray:
        return self.coords[-1]

    def length(self) -> float:
        return np.sum(_curve_norm(np.diff(self.coords, axis=0)))

    def __iter__(self) -> Iterator[tuple[np.floating, NDArray]]:
        yield from zip(self.param, self.coords)

    def __mul__(self, other) -> Self:
        rhs = self._get_bop_rhs(other)
        new_coords = self.coords * rhs
        return self.new(new_coords, self.param)

    def __truediv__(self, other) -> Self:
        rhs = self._get_bop_rhs(other)
        new_coords = self.coords / rhs
        return self.new(new_coords, self.param)

    def __add__(self, other) -> Self:
        rhs = self._get_bop_rhs(other)
        new_coords = self.coords + rhs
        return self.new(new_coords, self.param)

    def __sub__(self, other) -> Self:
        rhs = self._get_bop_rhs(other)
        new_coords = self.coords - rhs
        return self.new(new_coords, self.param)

    def __getitem__(self, key) -> Self:
        coords = self.coords[key]
        param = self.param[key]
        return self.new(coords, param)

    def _warn_not_unit(self) -> None:
        avg_speed = self.dot().norm().mean()
        msg = (
            f"Curve is not unit speed (average speed = {avg_speed:.2f}).\n"
            "Increase the number of points along the curve."
        )
        warnings.warn(msg)

    def _get_bop_rhs(self, other):
        if isinstance(other, Curve):
            if not np.all(self.param == other.param):
                msg = "Curves must have same parameterisation in order to be added."
                raise ValueError(msg)
            rhs = other.coords
        else:
            rhs = other
        return rhs


@dataclass(frozen=True)
class PlaneCurve(Curve):
    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            msg = "Coords of curve must have shape (N, 2)"
            raise ValueError(msg)
        if self.coords.shape[1] != 2:
            msg = "Plane curves must have D=2"
            raise ValueError(msg)
        if self.param.ndim != 1:
            msg = "Param of curve must have shape (N,)"
            raise ValueError(msg)
        if self.coords.shape[0] != self.param.shape[0]:
            msg = "Coords and param must have the same length"
            raise ValueError(msg)

    @classmethod
    def from_angle(cls, angle: NDArray, t: NDArray) -> Self:
        x = _integrate_from_zero(np.cos(angle), x=t)
        y = _integrate_from_zero(np.sin(angle), x=t)
        return cls(np.c_[x, y], t)

    @property
    def x(self) -> NDArray:
        return self.coords[:, 0]

    @property
    def y(self) -> NDArray:
        return self.coords[:, 1]

    def turning_angle(self) -> NDArray:
        tangent = self.tangent()
        tx, ty = tangent.coords[:, 0], tangent.coords[:, 1]
        return np.arctan2(ty, tx)

    def signed_normal(self) -> Self:
        tangent = self.tangent()
        rotated = tangent.coords @ ROT_90_DEG.T
        return self.new(rotated, self.param)

    def signed_curvature(self) -> NDArray:
        dot = self.dot()
        t = dot.normalise()
        ns_coords = t.coords @ ROT_90_DEG.T
        dtdt_coords = t.dot().coords
        signed_curvature = np.vecdot(dtdt_coords, ns_coords) / dot.norm()
        return signed_curvature

    def intersect(self, other) -> NDArray:
        return np.c_[intersection(self.x, self.y, other.x, other.y)]

    def intersect_coords(self, coords) -> NDArray:
        return np.c_[intersection(self.x, self.y, coords[:, 0], coords[:, 1])]


@dataclass(frozen=True)
class SpaceCurve(Curve):
    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            msg = "Coords of curve must have shape (N, 3)"
            raise ValueError(msg)
        if self.coords.shape[1] != 3:
            msg = "Space curves must have D=3"
            raise ValueError(msg)
        if self.param.ndim != 1:
            msg = "Param of curve must have shape (N,)"
            raise ValueError(msg)
        if self.coords.shape[0] != self.param.shape[0]:
            msg = "Coords and param must have the same length"
            raise ValueError(msg)

    def curvature(self) -> NDArray:
        # Can improve curvature calculation slightly in R^3
        dot = self.dot()
        ddot = dot.dot()
        A = _curve_norm(np.cross(ddot.coords, dot.coords))
        B = dot.norm() ** 3
        return A / B

    def normal(self) -> Self:
        _, normal = self.normal_and_tangent()
        return normal

    def binormal(self) -> Self:
        _, _, binormal = self.local_basis()
        return binormal

    def torsion(self) -> NDArray:
        _, n, b = self.local_basis()
        return -np.vecdot(b.dot().coords, n.coords)

    def normal_and_tangent(self) -> tuple[Self, Self]:
        t = self.tangent()
        n = t.dot().normalise()
        return t, n

    def local_basis(self) -> tuple[Self, Self, Self]:
        t, n = self.normal_and_tangent()
        b_coords = np.cross(t.coords, n.coords)
        b = self.new(b_coords, self.param)
        return t, n, b


def _curve_deriv(vectors: ArrayLike, param: ArrayLike) -> NDArray:
    return np.gradient(vectors, param, axis=0)


def _curve_norm(vectors: ArrayLike) -> NDArray:
    return np.linalg.norm(vectors, axis=1)


def _integrate_from_zero(y: ArrayLike, x: ArrayLike) -> NDArray:
    return cumulative_simpson(y, x=x, initial=0)


def plot_plane_curve(curve: PlaneCurve, ax=None, *args, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    fig = ax.figure

    plt.plot(curve.coords[:, 0], curve.coords[:, 1], *args, **kwargs)

    return fig
