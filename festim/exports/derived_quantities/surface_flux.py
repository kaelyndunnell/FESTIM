from festim import SurfaceQuantity, k_B
import fenics as f
import numpy as np


class SurfaceFlux(SurfaceQuantity):
    def __init__(self, field, surface) -> None:
        """
        Object to compute the flux J of a field u through a surface
        J = integral(-prop * grad(u) . n ds)
        where prop is the property of the field (D, thermal conductivity, etc)
        u is the field
        n is the normal vector of the surface
        ds is the surface measure.

        Note: this will probably won't work correctly for axisymmetric meshes.
        Use only with cartesian coordinates.

        Args:
            field (str, int):  the field ("solute", 0, 1, "T", "retention")
            surface (int): the surface id
        """
        super().__init__(field=field, surface=surface)
        self.title = "Flux surface {}: {}".format(self.surface, self.field)

    @property
    def prop(self):
        field_to_prop = {
            "0": self.D,
            "solute": self.D,
            0: self.D,
            "T": self.thermal_cond,
        }
        return field_to_prop[self.field]

    def compute(self, soret=False):
        flux = f.assemble(
            self.prop * f.dot(f.grad(self.function), self.n) * self.ds(self.surface)
        )
        if soret and self.field in [0, "0", "solute"]:
            flux += f.assemble(
                self.prop
                * self.function
                * self.Q
                / (k_B * self.T**2)
                * f.dot(f.grad(self.T), self.n)
                * self.ds(self.surface)
            )
        return flux


class SurfaceFluxCylindrical(SurfaceFlux):
    def __init__(self, field, surface, azimuth_range=(0, 2 * np.pi)) -> None:
        super().__init__(field, surface)
        self.r = None
        self.azimuth_range = azimuth_range

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > 2 * np.pi:
            raise ValueError("Azimuthal range must be between 0 and pi")
        self._azimuth_range = value

    def compute(self, soret=False):
        if soret:
            raise NotImplementedError(
                "Soret effect not implemented for cylindrical coordinates"
            )

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaz = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaz[0]  # only care about r here

        # dS_z = r dr dtheta , assuming axisymmetry dS_z = theta r dr
        # dS_r = r dz dtheta , assuming axisymmetry dS_r = theta r dz
        # in both cases the expression with self.ds is the same
        # we assume full cylinder theta = 2 pi
        flux = f.assemble(
            self.prop
            * self.r
            * f.dot(f.grad(self.function), self.n)
            * self.ds(self.surface)
        )
        flux *= self.azimuth_range[1] - self.azimuth_range[0]
        return flux


class SurfaceFluxSpherical(SurfaceFlux):
    def __init__(
        self, field, surface, azimuth_range=(0, np.pi), polar_range=(-np.pi, np.pi)
    ) -> None:
        super().__init__(field, surface)
        self.r = None
        self.polar_range = polar_range
        self.azimuth_range = azimuth_range

    @property
    def polar_range(self):
        return self._polar_range

    @polar_range.setter
    def polar_range(self, value):
        if value[0] < -np.pi or value[1] > np.pi:
            raise ValueError("Polar range must be between - pi and pi")
        self._polar_range = value

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > np.pi:
            raise ValueError("Azimuthal range must be between 0 and pi")
        self._azimuth_range = value

    def compute(self, soret=False):
        if soret:
            raise NotImplementedError(
                "Soret effect not implemented for spherical coordinates"
            )

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaphi = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaphi[0]  # only care about r here

        # dS_r = r^2 sin(theta) dtheta dphi
        # integral(f dS_r) = integral(f r^2 sin(theta) dtheta dphi)
        #                  = (phi2 - phi1) * (-cos(theta2) + cos(theta1)) * f r^2
        flux = f.assemble(
            self.prop
            * self.r**2
            * f.dot(f.grad(self.function), self.n)
            * self.ds(self.surface)
        )
        flux *= (self.polar_range[1] - self.polar_range[0]) * (
            -np.cos(self.azimuth_range[1]) + np.cos(self.azimuth_range[0])
        )
        return flux
