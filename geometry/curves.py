from intersect import intersection
from dataclasses import dataclass, field
from typing import Generic, Iterator, Self, TypeVar
import warnings
from scipy.integrate import cumulative_simpson, simpson
from scipy.interpolate import make_interp_spline, BSpline
from functools import partial
from numpy.typing import ArrayLike, NDArray
import numpy as np


ROT_90_DEG = np.array([[0, -1], [1, 0]])


norm = partial(np.linalg.norm, axis=-1)
CLOSE_TOL = 1e-5


@dataclass(frozen=True)
class Curve:
    coords: NDArray  # Must be (N, D)
    param: NDArray  # Must be (N,)
    _property_cache: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def new(cls, coords, param) -> Self:
        return cls(coords, param)

    @classmethod
    def new_unit_speed(cls, coords) -> Self:
        param = np.arange(coords.shape[0])
        curve = cls(coords, param).reparameterise_unit()
        return curve

    def __post_init__(self) -> None:
        # Convert to numpy arrays and validate
        coords = np.asarray(self.coords, dtype=float)
        param = np.asarray(self.param, dtype=float)

        # Update the actual arrays in case conversion was needed
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "param", param)

        # Basic shape validation
        if coords.ndim != 2:
            raise ValueError("Coords of curve must have shape (N, D)")
        if param.ndim != 1:
            raise ValueError("Param of curve must have shape (N,)")
        if coords.shape[0] != param.shape[0]:
            raise ValueError(
                f"Coords and param must have the same length: "
                f"coords has {coords.shape[0]} points, param has {param.shape[0]} points"
            )

        # Check for empty arrays
        if coords.size == 0 or param.size == 0:
            raise ValueError("Coords and param cannot be empty")

        # Check for NaN/inf values
        if not np.isfinite(coords).all():
            raise ValueError("Coords contains NaN or infinite values")
        if not np.isfinite(param).all():
            raise ValueError("Param contains NaN or infinite values")

        # Check parameter monotonicity (warning only)
        if len(param) > 1 and not np.all(np.diff(param) >= 0):
            warnings.warn(
                "Parameter values are not monotonically increasing", UserWarning
            )

    def is_unit(self, atol: float = CLOSE_TOL) -> bool:
        """True if this curve has unit speed along its length.

        Args:
            atol: Absolute tolerance for unit speed check.
        """
        # Simple instance-based cache to avoid hashing issues with numpy arrays
        cache_key = f"is_unit_{atol}"

        if cache_key not in self._property_cache:
            speed = self.dot().norm()
            result = np.allclose(speed, 1.0, atol=atol)
            # Use object.__setattr__ to modify frozen dataclass
            self._property_cache[cache_key] = result

        return self._property_cache[cache_key]

    def is_normalised(self, atol: float = CLOSE_TOL) -> bool:
        """True if this curve has parameter from 0 to 1.

        Args:
            atol: Absolute tolerance for normalization check.
        """
        cache_key = f"is_normalised_{atol}"

        if cache_key not in self._property_cache:
            result = np.isclose(self.param[0], 0.0, atol=atol) and np.isclose(
                self.param[-1], 1.0, atol=atol
            )
            self._property_cache[cache_key] = result

        return self._property_cache[cache_key]

    @property
    def num_points(self) -> int:
        """Return the number of points in the curve."""
        return len(self.param)

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

    def tangent(self, atol: float = CLOSE_TOL) -> Self:
        """Return a new curve which is the unit tangent vector of this curve.

        Args:
            atol: Absolute tolerance for unit speed check.
        """
        dot = self.dot()
        if self.is_unit(atol):
            return dot
        speed = dot.norm()[:, np.newaxis]
        # Handle zero speed (stationary points)
        speed = np.where(speed > np.finfo(float).eps, speed, 1.0)
        return self.new(dot.coords / speed, self.param)

    def curvature(self, atol: float = CLOSE_TOL) -> NDArray:
        """Return the curvature at each point along the curve.

        Args:
            atol: Absolute tolerance for unit speed check.
        """
        if self.is_unit(atol):
            return _curve_norm(self.ddot().coords)
        dot = self.dot()
        ddot = dot.dot()
        A1 = ddot.coords * np.vecdot(dot.coords, dot.coords)[:, np.newaxis]
        A2 = dot.coords * np.vecdot(dot.coords, ddot.coords)[:, np.newaxis]
        A = _curve_norm(A1 - A2)
        B = dot.norm() ** 4
        # Handle near-zero denominator for numerical stability
        B = np.where(B > np.finfo(float).eps ** 2, B, np.inf)
        return A / B

    def arc_length(self, atol: float = CLOSE_TOL) -> NDArray:
        """Return the arc-length of each point along the curve, starting at zero.

        Args:
            atol: Absolute tolerance for unit speed check.
        """
        if self.is_unit(atol):
            return self.param
        delta = _curve_norm(np.diff(self.coords, axis=0))
        return np.r_[0, np.cumsum(delta)]
        # speed = self.dot().norm()
        # return _integrate_from_zero(speed, self.param)

    def interpolate(self, t: ArrayLike, *args, **kwargs) -> Self:
        """Return a new curve formed by interpolating coordinates of this curve
        at the parameter values `t`.

        Additional arguments are passed to the Scipy function `make_interp_spline`.
        """
        t = np.asarray(t)

        # Check bounds for extrapolation warning
        t_min, t_max = self.param[0], self.param[-1]
        if np.any(t < t_min) or np.any(t > t_max):
            warnings.warn(
                f"Interpolation parameter outside bounds [{t_min:.3f}, {t_max:.3f}]. "
                "Extrapolation may be inaccurate.",
                UserWarning,
            )

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

    def reparameterise_unit(self, strict: bool = False) -> Self:
        """Return a new curve with unit speed parameterisation.

        Args:
            strict: If True, raise exception if unit speed cannot be achieved.
                   If False, issue warning only.
        """
        s = self.arc_length()
        new_curve = self.reparameterise(s)
        if not new_curve.is_unit():
            avg_speed = new_curve.dot().norm().mean()
            if strict:
                raise ValueError(
                    f"Failed to create unit speed curve (avg speed = {avg_speed:.3f}). "
                    "Consider increasing point density."
                )
            else:
                warnings.warn(
                    f"Curve is not unit speed (avg speed = {avg_speed:.3f}). "
                    "Consider increasing point density.",
                    UserWarning,
                )
        return new_curve

    def normalise(self) -> Self:
        """Return a new curve which has parameter between 0 and 1."""
        p0, pN = self.param[[0, -1]]
        new_param = (self.param - p0) / (pN - p0)
        return self.reparameterise(new_param)

    def start(self) -> NDArray:
        """Return the first point of the curve."""
        return self.coords[0]

    def end(self) -> NDArray:
        """Return the last point of the curve."""
        return self.coords[-1]

    def length(self) -> float:
        """Return the total arc length of the curve."""
        return np.sum(_curve_norm(np.diff(self.coords, axis=0)))

    def tangent_vectors(self) -> NDArray:
        """Return raw tangent vectors without curve wrapper."""
        return self.tangent().coords

    def curvature_curve(self) -> Self:
        """Return curvature as a 1D curve for consistent chaining."""
        kappa = self.curvature()
        return self.new(kappa[:, np.newaxis], self.param)

    @property
    def is_closed(self) -> bool:
        """Check if the curve is closed (start and end points are the same)."""
        return np.allclose(self.start(), self.end(), atol=CLOSE_TOL)

    def __iter__(self) -> Iterator[tuple[np.floating, NDArray]]:
        yield from zip(self.param, self.coords)

    def __repr__(self) -> str:
        n_points = len(self.coords)
        dim = self.coords.shape[1]
        param_range = f"[{self.param[0]:.3f}, {self.param[-1]:.3f}]"
        return f"Curve({n_points} points, {dim}D, param={param_range})"

    def __str__(self) -> str:
        return self.__repr__()

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

    def _get_bop_rhs(self, other, rtol: float = 1e-10):
        """Get right-hand side for binary operations.

        Args:
            rtol: Relative tolerance for parameter comparison.
        """
        if isinstance(other, Curve):
            if not np.allclose(self.param, other.param, rtol=rtol):
                raise ValueError(
                    "Curves must have same parameterisation for binary operations."
                )
            rhs = other.coords
        else:
            rhs = other
        return rhs


@dataclass(frozen=True)
class PlaneCurve(Curve):
    def __post_init__(self) -> None:
        # Call parent validation first
        super().__post_init__()

        # Add plane-specific validation
        if self.coords.shape[1] != 2:
            raise ValueError("Plane curves must have D=2")

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

    def plot(self, ax=None, *args, **kwargs):
        """Plot the plane curve.

        Args:
            ax: Matplotlib axes object. If None, creates new figure.
            *args, **kwargs: Additional arguments passed to matplotlib plot.

        Returns:
            The axes object used for plotting.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.coords[:, 0], self.coords[:, 1], *args, **kwargs)
        return ax


@dataclass(frozen=True)
class SpaceCurve(Curve):
    def __post_init__(self) -> None:
        # Call parent validation first
        super().__post_init__()

        # Add space-specific validation
        if self.coords.shape[1] != 3:
            raise ValueError("Space curves must have D=3")

    def curvature(self, atol: float = CLOSE_TOL) -> NDArray:
        """Return the curvature at each point along the curve (optimized for 3D).

        Args:
            atol: Absolute tolerance for unit speed check (unused in 3D formula).
        """
        # Can improve curvature calculation slightly in R^3
        # Note: atol parameter maintained for API consistency but unused here
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
    """Legacy function - use curve.plot() method instead."""
    return curve.plot(ax, *args, **kwargs)
