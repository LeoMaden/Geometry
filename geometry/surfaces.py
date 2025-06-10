from typing import Literal
from scipy.interpolate import interpn
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

    @classmethod
    def from_parameter_curves(
        cls, curves: list[PlaneCurve | SpaceCurve], v: ArrayLike
    ) -> "SurfacePatch":
        v = np.asarray(v)

        for c1, c2 in zip(curves[:-1], curves[1:]):
            if not np.all(c1.param == c2.param):
                msg = "Curves must be defined with the same parameterisation."
                raise ValueError(msg)
        if len(v) != len(curves):
            msg = "`curves` and `v` must be the same length."
            raise ValueError(msg)

        u = curves[0].param
        N, M = len(u), len(v)
        D = 2 if isinstance(curves[0], PlaneCurve) else 3
        coords = np.empty((N, M, D))

        for iv, c in enumerate(curves):
            coords[:, iv] = c.coords

        return SurfacePatch(coords, u, v)

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
