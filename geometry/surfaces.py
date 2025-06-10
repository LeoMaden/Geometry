from typing import Literal, Optional
from scipy.interpolate import interpn, CubicHermiteSpline
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray, ArrayLike

from geometry.curves import PlaneCurve, SpaceCurve


@dataclass
class SurfacePatch:
    """Represents a surface patch in R^2 or R^3 parameterised by a rectangular grid
    of `u` and `v` values.
    """

    coords: NDArray  # Shape (N, M, D) where D in [2, 3]
    u: NDArray  # Shape (N,)
    v: NDArray  # Shape (M,)

    def is_2d(self) -> bool:
        return self.coords.shape[2] == 2

    def is_3d(self) -> bool:
        return self.coords.shape[2] == 3

    def grad_u(self) -> "SurfacePatch":
        grad_u = np.gradient(self.coords, self.u, axis=0)
        return SurfacePatch(grad_u, self.u, self.v)

    def grad_v(self) -> "SurfacePatch":
        grad_v = np.gradient(self.coords, self.v, axis=1)
        return SurfacePatch(grad_v, self.u, self.v)

    def normal(self) -> "SurfacePatch":
        grad_u = self.grad_u()
        grad_v = self.grad_v()
        normal = np.cross(grad_u.coords, grad_v.coords)
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        return SurfacePatch(normal, self.u, self.v)

    def interpolate(self, u=None, v=None, **kwargs) -> "SurfacePatch":
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        coords = interpn(
            (self.u, self.v),
            self.coords,
            np.meshgrid(u, v, indexing="ij"),
            **kwargs,
        )
        return SurfacePatch(coords, u, v)

    def parameter_curves(
        self, const_axis: Literal["u", "v"]
    ) -> list[PlaneCurve | SpaceCurve]:
        if self.is_2d():
            cls = PlaneCurve
        else:
            cls = SpaceCurve

        nu, nv = self.coords.shape[:2]

        if const_axis == "v":
            return [cls(self.coords[:, j], self.u) for j in range(nv)]
        else:
            return [cls(self.coords[i, :], self.v) for i in range(nu)]

    def __add__(self, other) -> "SurfacePatch":
        if isinstance(other, SurfacePatch):
            rhs = other.coords
        else:
            rhs = other
        coords = self.coords + rhs
        return SurfacePatch(coords, self.u, self.v)

    def __sub__(self, other) -> "SurfacePatch":
        if isinstance(other, SurfacePatch):
            rhs = other.coords
        else:
            rhs = other
        coords = self.coords - rhs
        return SurfacePatch(coords, self.u, self.v)
