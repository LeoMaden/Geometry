from dataclasses import dataclass
from typing import Optional
from curves import PlaneCurve
from surfaces import SurfacePatch
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import CubicHermiteSpline


@dataclass
class Grid2D:
    left: Optional[PlaneCurve]  # Coords shape (M, 2)
    right: Optional[PlaneCurve]  # Coords shape (M, 2)
    top: Optional[PlaneCurve]  # Coords shape (N, 2)
    bot: Optional[PlaneCurve]  # Coords shape (N, 2)

    def __post_init__(self) -> None:
        if self.left:
            self.left = self.left.normalise()
        if self.right:
            self.right = self.right.normalise()
        if self.top:
            self.top = self.top.normalise()
        if self.bot:
            self.bot = self.bot.normalise()

    def projector_u(self, u_eval: NDArray) -> SurfacePatch:
        assert self.left is not None
        assert self.right is not None

        # TODO: Validate

        u = np.array([0, 1])
        v = self.left.param
        coords = np.stack((self.left.coords, self.right.coords), axis=0)
        return SurfacePatch(coords, u, v).interpolate(u=u_eval)

    def projector_v(self, v_eval: NDArray) -> SurfacePatch:
        assert self.top is not None
        assert self.bot is not None

        # TODO: Validate

        u = self.bot.param
        v = np.array([0, 1])
        coords = np.stack((self.bot.coords, self.top.coords), axis=1)
        return SurfacePatch(coords, u, v).interpolate(v=v_eval)

    def projector_uv(self, u_eval: NDArray, v_eval: NDArray) -> SurfacePatch:
        u = np.array([0, 1])
        v = np.array([0, 1])
        return SurfacePatch(self.corners(), u, v).interpolate(u=u_eval, v=v_eval)

    def corners(self) -> NDArray:
        if self.top and self.bot:
            return self._corners_top_bot()
        elif self.left and self.right:
            return self._corners_left_right()
        msg = "Must define left and right or top and bottom boundaries."
        raise ValueError(msg)

    def tfi(self, u_eval: NDArray, v_eval: NDArray) -> SurfacePatch:
        Pu = self.projector_u(u_eval).interpolate(v=v_eval)
        Pv = self.projector_v(v_eval).interpolate(u=u_eval)
        PuPv = self.projector_uv(u_eval, v_eval)
        return Pu + Pv - PuPv

    def hermite_u(self, u_eval: NDArray, amount: float = 1.0) -> SurfacePatch:
        assert self.left is not None
        assert self.right is not None

        # TODO: Validate

        u = np.array([0, 1])
        v = self.left.param
        coords = np.stack((self.left.coords, self.right.coords), axis=0)
        gradient = amount * np.stack(
            (self.left.signed_normal().coords, self.right.signed_normal().coords),
            axis=0,
        )

        spl = CubicHermiteSpline(u, coords, gradient, axis=0)
        new_coords = spl(u_eval)
        return SurfacePatch(new_coords, u_eval, v)

    def hermite_v(self, v_eval: NDArray, amount: float = 1.0) -> SurfacePatch:
        assert self.top is not None
        assert self.bot is not None

        # TODO: Validate

        u = self.bot.param
        v = np.array([0, 1])
        coords = np.stack((self.bot.coords, self.top.coords), axis=1)
        gradient = amount * np.stack(
            (self.bot.signed_normal().coords, self.top.signed_normal().coords), axis=1
        )

        spl = CubicHermiteSpline(v, coords, gradient, axis=1)
        new_coords = spl(v_eval)
        return SurfacePatch(new_coords, u, v_eval)

    def _corners_left_right(self) -> NDArray:
        assert self.left is not None
        assert self.right is not None
        return np.array(
            [
                [self.left.coords[0], self.left.coords[-1]],
                [self.right.coords[0], self.right.coords[-1]],
            ]
        )

    def _corners_top_bot(self) -> NDArray:
        assert self.top is not None
        assert self.bot is not None
        return np.array(
            [
                [self.bot.coords[0], self.top.coords[0]],
                [self.bot.coords[-1], self.top.coords[-1]],
            ]
        )
