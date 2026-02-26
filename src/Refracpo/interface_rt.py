import numpy as np
import torch as T

from .EMtools import poyntingVector
from .FresnelCoeff import Fresnel_coeffi
from .vecops import Vector, cross, dot, magnitude, normalized


def _as_vector(v):
    """Ensure return type is Vector with x/y/z components."""
    if isinstance(v, Vector):
        return v
    return Vector(v[0], v[1], v[2])


def calc_reflect_transmit_fields(E, H, n_hat, n1, n2, AR=None):
    """
    Unified interface for non-AR and AR-coated interfaces.

    Parameters
    ----------
    E, H : Vector
        Incident electric and magnetic fields.
    n_hat : Vector
        Interface normal (nominally pointing from medium n1 to n2).
    n1, n2 : float
        Refractive indices of incident/transmission media.
    AR : callable or None
        - None: use Fresnel_coeffi(n1, n2, cos(theta_i)).
        - callable: use AR(theta_i) in radians, returns (t_p, t_s, r_p, r_s).

    Returns
    -------
    E_t, H_t, E_r, H_r, t_p, t_s, r_p, r_s
    """
    # Propagation direction approximated by normalized Poynting vector
    S_i = poyntingVector(E, H)
    k_i = normalized(S_i)

    # Ensure normal orientation gives positive incidence cosine
    cos_i = dot(n_hat, k_i)
    cos_i_np = cos_i.detach().cpu().numpy() if isinstance(cos_i, T.Tensor) else np.asarray(cos_i)
    flip_mask = cos_i_np < 0
    if np.any(flip_mask):
        # Flip normals point-wise instead of global flipping.
        # This avoids corrupting points that already have correct normal orientation.
        if isinstance(n_hat.x, T.Tensor):
            mask_t = T.as_tensor(flip_mask, dtype=T.bool, device=n_hat.x.device)
            n_hat = Vector(
                T.where(mask_t, -n_hat.x, n_hat.x),
                T.where(mask_t, -n_hat.y, n_hat.y),
                T.where(mask_t, -n_hat.z, n_hat.z),
            )
        else:
            n_hat = Vector(
                np.where(flip_mask, -n_hat.x, n_hat.x),
                np.where(flip_mask, -n_hat.y, n_hat.y),
                np.where(flip_mask, -n_hat.z, n_hat.z),
            )
        cos_i = dot(n_hat, k_i)
        cos_i_np = cos_i.detach().cpu().numpy() if isinstance(cos_i, T.Tensor) else np.asarray(cos_i)

    # Reflected direction
    k_r = k_i - 2.0 * dot(k_i, n_hat) * n_hat

    # Snell relation for transmitted direction
    sin_i2 = np.maximum(0.0, 1.0 - cos_i_np**2)
    sin_t2 = (n1 / n2) ** 2 * sin_i2
    tir = np.abs(sin_t2) >= 1.0
    sin_t2_clip = np.clip(sin_t2, 0.0, 1.0)
    cos_t = np.sqrt(1.0 - sin_t2_clip)

    k_t = (n1 / n2) * (k_i - cos_i * n_hat) + cos_t * n_hat

    # Build s/p basis
    s = cross(k_i, n_hat)
    s_mag_np = np.asarray(magnitude(s))
    degenerate = s_mag_np < 1e-15
    if np.any(degenerate):
        ones = np.ones_like(s_mag_np)
        zeros = np.zeros_like(s_mag_np)
        ex_v = Vector(ones, zeros, zeros)
        ey_v = Vector(zeros, ones, zeros)
        s_alt = cross(n_hat, ex_v)
        s_alt_mag_np = np.asarray(magnitude(s_alt))
        if np.any(s_alt_mag_np < 1e-15):
            s_alt = cross(n_hat, ey_v)
        s = s_alt
    s = normalized(s)

    p_i = normalized(cross(s, k_i))
    p_r = normalized(cross(s, k_r))
    p_t = normalized(cross(s, k_t))

    # Coefficients: plain Fresnel or AR lookup
    if AR is None:
        t_p, t_s, r_p, r_s = Fresnel_coeffi(n1, n2, cos_i_np)
    else:
        theta_i = np.arccos(np.clip(np.abs(cos_i_np), -1.0, 1.0))
        t_p, t_s, r_p, r_s = AR(theta_i)

    # Decompose incident field on (s, p_i)
    E_i_s = dot(E, s)
    E_i_p = dot(E, p_i)

    # Reconstruct reflected/transmitted E fields
    E_r = (r_s * E_i_s) * s + (r_p * E_i_p) * p_r
    E_t = (t_s * E_i_s) * s + (t_p * E_i_p) * p_t

    # Plane-wave relation: H = n * (k x E)
    H_r = n1 * cross(k_r, E_r)
    H_t = n2 * cross(k_t, E_t)

    # TIR branch: suppress transmitted propagating field in this simplified model
    if np.any(tir):
        if isinstance(E_t.x, T.Tensor):
            tir_mask_t = T.as_tensor(tir, dtype=T.bool, device=E_t.x.device)
            zero_ex = T.zeros_like(E_t.x)
            zero_hx = T.zeros_like(H_t.x)
            E_t = Vector(
                T.where(tir_mask_t, zero_ex, E_t.x),
                T.where(tir_mask_t, T.zeros_like(E_t.y), E_t.y),
                T.where(tir_mask_t, T.zeros_like(E_t.z), E_t.z),
            )
            H_t = Vector(
                T.where(tir_mask_t, zero_hx, H_t.x),
                T.where(tir_mask_t, T.zeros_like(H_t.y), H_t.y),
                T.where(tir_mask_t, T.zeros_like(H_t.z), H_t.z),
            )
        else:
            E_t = Vector(np.where(tir, 0, E_t.x), np.where(tir, 0, E_t.y), np.where(tir, 0, E_t.z))
            H_t = Vector(np.where(tir, 0, H_t.x), np.where(tir, 0, H_t.y), np.where(tir, 0, H_t.z))

    # Keep interface consistent: returned fields are always Vector objects.
    E_t = _as_vector(E_t)
    H_t = _as_vector(H_t)
    E_r = _as_vector(E_r)
    H_r = _as_vector(H_r)

    return E_t, H_t, E_r, H_r, t_p, t_s, r_p, r_s
