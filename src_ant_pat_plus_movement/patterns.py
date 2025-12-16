import sionna.rt
import numpy as np
import pandas as pd
import drjit as dr
import mitsuba as mi

from sionna.rt import AntennaPattern
from scipy.interpolate import RegularGridInterpolator
from typing import Callable


def sph_to_cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def rotation_matrix(angles):
    """
    Computes the rotation matrix from ZYX Euler angles.

    Parameters
    ----------
    angles : array-like of shape (3,)
        Rotation angles [alpha, beta, gamma] in radians.
        alpha : rotation about Z
        beta  : rotation about Y
        gamma : rotation about X

    Returns
    -------
    rot_mat : ndarray of shape (3, 3)
        Rotation matrix.
    """

    a, b, c = angles  # alpha, beta, gamma

    sin_a, cos_a = np.sin(a), np.cos(a)
    sin_b, cos_b = np.sin(b), np.cos(b)
    sin_c, cos_c = np.sin(c), np.cos(c)

    r_11 = cos_a * cos_b
    r_12 = cos_a * sin_b * sin_c - sin_a * cos_c
    r_13 = cos_a * sin_b * cos_c + sin_a * sin_c

    r_21 = sin_a * cos_b
    r_22 = sin_a * sin_b * sin_c + cos_a * cos_c
    r_23 = sin_a * sin_b * cos_c - cos_a * sin_c

    r_31 = -sin_b
    r_32 = cos_b * sin_c
    r_33 = cos_b * cos_c

    rot_mat = np.array([[r_11, r_12, r_13], [r_21, r_22, r_23], [r_31, r_32, r_33]])

    return rot_mat


def cart_to_sph(v):
    # v is the (3, N) matrix, where N is the number of rays
    x, y, z = v[0, :], v[1, :], v[2, :]

    # FIX: Calculate the norm for each ray (column) along axis 0.
    r = np.linalg.norm(v, axis=0)  # <--- THIS IS CRITICAL AND MUST BE IN YOUR CODE

    z_over_r = np.clip(z / r, -1.0, 1.0)
    theta = np.arccos(z_over_r)
    phi = np.arctan2(y, x)

    # We apply the phi wrap here for proper [0, 2pi] indexing
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return theta, phi


def adjust_angles(theta, phi):
    v = sph_to_cart(theta, phi)
    rot_mat = rotation_matrix(np.array([0, -dr.pi / 2, 0]))  # rotate_z_to_x(v)
    v_rot = rot_mat @ v
    theta_r, phi_r = cart_to_sph(v_rot)

    return theta_r, phi_r


""" custom patterns for sionna based on measurements """


# antenna pattern based on measurements
class MeasuredPattern(AntennaPattern):
    """
    Custom antenna pattern based on CSV data.
    The CSV must contain columns:
    'Phi[deg]', 'Theta[deg]', 'mag(rERHCP)[mV]', 'ang_deg(rERHCP)[deg]'
    """

    def __init__(self, csv_path, normalize=False):
        super().__init__()

        # ---- Load CSV ----
        self.path = csv_path
        df = pd.read_csv(csv_path)
        phi_deg = df["Phi[deg]"].values  # degrees
        theta_deg = df["Theta[deg]"].values  # degrees
        mag = df["mag(rERHCP)[mV]"].values / 1000  # convert mV to V
        # print(f'max mag in MeasuredPattern: {np.max(mag)*1000} mV')
        ang_deg = df["ang_deg(rERHCP)[deg]"].values

        # ---- Convert to radians and complex field ----
        phi = np.deg2rad(phi_deg)
        theta = np.deg2rad(theta_deg)
        field = mag * np.exp(1j * np.deg2rad(ang_deg))

        # Optional normalization (to make max field = 1)
        if normalize:
            field = field / np.max(np.abs(field))

        # ---- Build regular grids for interpolation ----
        # (assuming uniform or nearly-uniform sampling)
        phi_unique = np.sort(np.unique(phi))
        theta_unique = np.sort(np.unique(theta))

        # reshape to 2D grid [len(theta), len(phi)]
        n_theta = len(theta_unique)
        n_phi = len(phi_unique)
        field_grid = field.reshape(n_theta, n_phi)

        # Interpolator over (theta, phi)
        self._interp = RegularGridInterpolator(
            (theta_unique, phi_unique), field_grid, bounds_error=False, fill_value=1e-8
        )

    @property
    def patterns(self):
        def f(theta, phi):
            # Convert TF tensors to numpy arrays for interpolation
            theta_np = theta.numpy()
            phi_np = phi.numpy()

            # adjust radiation pattern orientation (pointing in z direction to pointing in x direction)
            theta_np, phi_np = adjust_angles(theta_np, phi_np)

            # Stack and interpolate
            pts = np.stack([theta_np, phi_np], axis=-1)
            field_np = self._interp(pts)

            field_np = np.asarray(field_np, dtype=np.complex64)
            E_theta_np = field_np / np.sqrt(2)
            E_phi_np = 1j * field_np / np.sqrt(2)

            # Convert to Dr.Jit type
            c_theta = mi.Complex2f(E_theta_np.real, E_theta_np.imag)
            c_phi = mi.Complex2f(E_phi_np.real, E_phi_np.imag)
            return c_theta, c_phi

        # Single polarization (RHCP)
        return [f]

# --- Custom Pattern Function using Dr.Jit/Mitsuba ---
def v_tr38901_pattern(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
    r"""
    Vertically polarized measured antenna pattern using closest point.
    """
 
    # # builtin dipole
    # k = dr.sqrt(1.5)
    # c_theta = dr.abs(k*dr.sin(theta))
 
    # return mi.Complex2f(c_theta, 0)
 
 
    # builtin v_tr38901_pattern
    # Wrap phi to [-PI,PI]
    phi = phi+dr.pi
    phi -= dr.floor(phi/(2.*dr.pi))*2.*dr.pi
    phi -= dr.pi
 
    # Zenith pattern
    theta_3db = phi_3db = 65./180.*dr.pi
    a_max = sla_v = 30.
    g_e_max = 20.
    a_v = -dr.min([12.*((theta-dr.pi/2.)/theta_3db)**2, sla_v])
    a_h = -dr.min([12.*(phi/phi_3db)**2, a_max])
    a_db = -dr.min([-(a_v + a_h), a_max]) + g_e_max
    a = dr.power(10., a_db/10.)
    c_theta = dr.sqrt(a)
 
    return mi.Complex2f(c_theta, 0)
 
    # Register all available antenna patterns
def create_factory(name: str) -> Callable[[str, str], sionna.rt.antenna_pattern.PolarizedAntennaPattern]:
    r"""Create a factory method for the instantiation of polarized antenna
    patterns
 
    Note that there must be a vertical antenna pattern function with name
    "v_{s}_pattern" which is used.
 
    :param name: Name under which to register the factory method
    :returns: Callable creating an instance of PolarizedAntennaPattern
    """
    def f(*, polarization, polarization_model="tr38901_2"):
        return sionna.rt.antenna_pattern.PolarizedAntennaPattern(
                                v_pattern=globals()["v_" + name + "_pattern"],
                                polarization=polarization,
                                polarization_model=polarization_model)
    return f
 
for s in ["tr38901"]:
    sionna.rt.antenna_pattern.register_antenna_pattern(s, create_factory(s))