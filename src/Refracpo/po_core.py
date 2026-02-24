"""
Physical Optics (PO) Core Calculation Module
=============================================

This module contains the fundamental electromagnetic field calculations
for Physical Optics analysis. It implements the integral
formulation for computing near-field and far-field radiation from surface currents.

The core physics:
    Based on the physicla optics approximation, electromagnetic fields can be
    computed by integrating over the radiation from surface electric and
    magnetic current distributions on an aperture or scatterer.

Key References:
    - Physical optics approximation
    - Vector potential formulation (A, F potentials)
    - Near-field electromagnetic propagation
    - Far-field radiation pattern.

Public API Functions:
    po_integrate_surface_currents_je: Integrate E-field only with Vector objects
    po_integrate_surface_currents: Integrate E-field and M-field with Vector objects

Internal JIT Functions (do not call directly):
    _po_integrate_surface_currents_je_jit: Core JIT-compiled calculation (Je only)
    _po_integrate_surface_currents_jit: Core JIT-compiled calculation (Je + Jm)
"""

import numpy as np
from numba import njit, prange
from .vecops import Vector
from . import c, mu, epsilon, Z0


def po_integrate_surface_currents_je(r_src,
                                     r_obs,
                                     N,
                                     ds,
                                     Je,
                                     k,
                                     Z=Z0):
    """
    Compute EM fields from surface electric current density (Jm = 0) using Vector objects.
    
    This is the user-facing wrapper that accepts Vector objects for source and 
    observation point positions, automatically handling NumPy/PyTorch backend conversion.
    
    Parameters:
    -----------
    r_src : Vector
        Source surface position vectors (3, N_source)
    r_obs : Vector
        Observation surface position vectors (3, N_obs)
    N : ndarray
        Jacobian factor for surface projection to 2D integration plane
        (accounts for surface orientation and projection geometry)
        Shape: (N_source,)
    ds : ndarray
        Differential area × sampling quadrature weights
        (combines element area with quadrature weights e.g., from Gaussian quadrature)
        Shape: (N_source,)
    Je : ndarray or Vector
        Electric surface current density
        Shape: (3, N_source) containing [Jex, Jey, Jez] components
    k : float
        Wave number (k = 2π/λ)
    Z : float, optional
        Impedance of the space (default: module constant is free space impedance)
    
    Returns:
    --------
    E_field : Vector
        Electric field at observation points (3, N_obs)
    H_field : Vector
        Magnetic field at observation points (3, N_obs)
    """
    if Z is None:
        Z = Z0*1.0
    
    # Extract coordinates from Vector objects
    x1 = r_src.x.ravel()
    y1 = r_src.y.ravel()
    z1 = r_src.z.ravel()
    
    x2 = r_obs.x.ravel()
    y2 = r_obs.y.ravel()
    z2 = r_obs.z.ravel()
    
    # Convert Je to NumPy if it's a Vector object
    if isinstance(Je, Vector):
        Je = np.vstack([Je.x, Je.y, Je.z])
    else:
        Je = np.atleast_2d(Je)
    
    # Call the JIT-compiled core function
    Field_E_x, Field_E_y, Field_E_z, Field_H_x, Field_H_y, Field_H_z = \
        _po_integrate_surface_currents_je_jit(x1, y1, z1, x2, y2, z2, N, ds, Je, k, Z)
    
    # Return as Vector objects
    E_field = Vector(Field_E_x, Field_E_y, Field_E_z)
    H_field = Vector(Field_H_x, Field_H_y, Field_H_z)
    
    return E_field, H_field


def po_integrate_surface_currents(r_src, r_obs, N, ds, Je, Jm, k, Z=None):
    """
    Compute EM fields from surface electric AND magnetic current densities using Vector objects.
    
    This is the user-facing wrapper that accepts Vector objects for source and 
    observation point positions, automatically handling NumPy/PyTorch backend conversion.
    
    Parameters:
    -----------
    r_src : Vector
        Source surface position vectors (3, N_source)
    r_obs : Vector
        Observation surface position vectors (3, N_obs)
    N : ndarray
        Jacobian factor for surface projection to 2D integration plane
        (accounts for surface orientation and projection geometry)
        Shape: (N_source,)
    ds : ndarray
        Differential area × sampling quadrature weights
        (combines element area with quadrature weights e.g., from Gaussian quadrature)
        Shape: (N_source,)
    Je : ndarray or Vector
        Electric surface current density (shape: (3, N_source))
    Jm : ndarray or Vector
        Magnetic surface current density (shape: (3, N_source))
    k : float
        Wave number (k = 2π/λ)
    Z : float, optional
        Impedance of free space (default: module constant)
    
    Returns:
    --------
    E_field : Vector
        Electric field at observation points (3, N_obs)
    H_field : Vector
        Magnetic field at observation points (3, N_obs)
    """
    if Z is None:
        from . import Z0 as Z0_const
        Z = Z0_const
    
    # Extract coordinates from Vector objects
    x1 = r_src.x.ravel()
    y1 = r_src.y.ravel()
    z1 = r_src.z.ravel()
    
    x2 = r_obs.x.ravel()
    y2 = r_obs.y.ravel()
    z2 = r_obs.z.ravel()
    
    # Convert Je, Jm to NumPy if they're Vector objects
    if isinstance(Je, Vector):
        Je = np.vstack([Je.x, Je.y, Je.z])
    else:
        Je = np.atleast_2d(Je)
    
    if isinstance(Jm, Vector):
        Jm = np.vstack([Jm.x, Jm.y, Jm.z])
    else:
        Jm = np.atleast_2d(Jm)
    
    # Call the JIT-compiled core function
    Field_E_x, Field_E_y, Field_E_z, Field_H_x, Field_H_y, Field_H_z = \
        _po_integrate_surface_currents_jit(x1, y1, z1, x2, y2, z2, N, ds, Je, Jm, k, Z)
    
    # Return as Vector objects
    E_field = Vector(Field_E_x, Field_E_y, Field_E_z)
    H_field = Vector(Field_H_x, Field_H_y, Field_H_z)
    
    return E_field, H_field


@njit(parallel=True)
def _po_integrate_surface_currents_je_jit(x1, y1, z1, x2, y2, z2, N, ds, Je, k, Z):
    """
    Compute EM fields from surface electric current density (Jm = 0).
    
    This function implements the core Physical Optics integral for the case
    where only electric surface currents are present (e.g., on a PEC surface
    with normal incident E-field).
    
    Mathematical Formulation:
    -------------------------
    The electric field is computed from the electric current distribution:
    
        E = ∫ G(k) · [k² Je(1j/kr - 1/kr² + 1j/kr³) 
                      + (Je·R̂) R̂ (-1j/kr + 3/kr² - 3j/kr³)]
            × N(r') ds' dω
    
    The magnetic field is:
    
        H = (1/Zf) ∫ k² G(k) R̂ × Je (1 - 1j·kr)/(kr²) × N(r') ds' dω
    
    Where:
        - R = r - r' is the vector from source to observation point
        - r = |R| is the distance
        - kr = k·r (normalized distance)
        - R̂ = R/r is the unit direction vector
        - G(k) = exp(j·kr)/(4π)
        - Zf = free space impedance
    
    Parameters:
    -----------
    x1, y1, z1 : ndarray
        Source surface coordinates (shape: (N_source,))
    x2, y2, z2 : ndarray
        Observation surface coordinates (shape: (N_obs,))
    N : ndarray
        Jacobian factor for surface projection to 2D integration plane
        (accounts for surface orientation and projection geometry)
        Shape: (N_source,)
    ds : ndarray
        Differential area × sampling quadrature weights
        (combines element area with quadrature weights e.g., from Gaussian quadrature)
        Shape: (N_source,)
    Je : ndarray
        Electric surface current density matrix
        Shape: (3, N_source) containing [Jex, Jey, Jez] components
    k : float
        Wave number (k = 2π/λ)
    Z0 : float
        Impedance of free space (~377 Ohm)
    
    Returns:
    --------
    Field_E_x, Field_E_y, Field_E_z : ndarray
        Electric field components at observation points
        Shape: (N_obs,), dtype: complex128
    Field_H_x, Field_H_y, Field_H_z : ndarray
        Magnetic field components at observation points
        Shape: (N_obs,), dtype: complex128
    
    Notes:
    ------
    1. This function is JIT-compiled with Numba for performance
    2. Parallel execution is enabled (thread-based parallelization)
    3. Numerical stability: uses explicit 1/kr rather than 1/r to avoid
       cancellation errors for kr ≈ 0
    4. The Green's function exp(jkr)/(4π) is factored into amplitudes
    
    Algorithm Steps:
    1. For each observation point i:
        a. Compute vector R from all source points to observation point i
        b. Compute distance r = |R|
        c. Compute normalized distance kr and its powers
        d. Compute Green's function phase: exp(-jkr) = exp(-jk|r|)
        e. Compute electric field contribution using vector potential formula
        f. Compute magnetic field contribution as curl of vector potential
        g. Sum contributions from all source points with weighting N·ds
        h. Apply final amplitude scaling factor Z0/(4π) or 1/(4π)
    
    Example:
    --------
    >>> # Simple case: line current source
    >>> x1 = np.linspace(-0.5, 0.5, 101)
    >>> y1 = np.zeros_like(x1)
    >>> z1 = np.zeros_like(x1)
    >>> x2 = np.array([0.0])
    >>> y2 = np.array([0.1])
    >>> z2 = np.array([0.0])
    >>> N = np.ones(101)
    >>> ds = np.ones(101) * 0.01
    >>> Je = np.vstack([np.ones(101), np.zeros(101), np.zeros(101)])
    >>> k = 2 * np.pi  # Unit wavelength
    >>> Z0 = 377.0
    >>> Ex, Ey, Ez, Hx, Hy, Hz = _po_integrate_surface_currents_je(
    ...     x1, y1, z1, x2, y2, z2, N, ds, Je, k, Z0)
    """
    
    Field_E_x = np.zeros(x2.shape, dtype=np.complex128)
    Field_E_y = np.zeros(x2.shape, dtype=np.complex128)
    Field_E_z = np.zeros(x2.shape, dtype=np.complex128)
    Field_H_x = np.zeros(x2.shape, dtype=np.complex128)
    Field_H_y = np.zeros(x2.shape, dtype=np.complex128)
    Field_H_z = np.zeros(x2.shape, dtype=np.complex128)
    
    # Iterate over all observation points (parallel)
    for i in prange(x2.size):
        # Step 1: Compute relative position vector R from source to observation point
        R = np.zeros((3, x1.size))
        R[0, ...] = x2[i] - x1.ravel()
        R[1, ...] = y2[i] - y1.ravel()
        R[2, ...] = z2[i] - z1.ravel()
        
        # Step 2: Compute distance and normalized wave distance
        r = np.sqrt(np.sum(R**2, axis=0))
        kr = k * r  # kr = k·r (normalized distance)
        kr2 = kr**2  # (kr)²
        kr3 = kr**3  # (kr)³
        kr2_inv = 1.0 / kr2  # 1/(kr²)
        kr3_inv = 1.0 / kr3  # 1/(kr³)
        
        # Step 3: Compute Green's function phase
        # G(k,r) = exp(jkr)/(4π), we use phase = -kr for exp(jkr)
        phase = -kr  # exp(-1j*kr) comes from convention
        
        # Step 4: Compute electric field from vector potential
        # ee represents the integrand for E-field calculation
        ee = np.exp(1j*phase) * k**2 * (
            Je * (1j/kr - kr2_inv + 1j*kr3_inv) +
            np.sum(Je*R/r, axis=0) * R/r * (-1j/kr + 3*kr2_inv - 3j*kr3_inv)
        )
        Ee = np.sum(ee * N * ds, axis=1)
        
        # Step 5: Compute magnetic field from vector potential curl
        # H = (1/μ₀) ∇ × A, with vector potential A
        he = np.exp(1j*phase) * k**2
        he1 = R/r * kr2_inv * (1 - 1j*kr)  # R̂ × (1 - j·kr)/(kr²)
        
        # Cross product: ∇ × Je
        he2 = np.zeros((3, x1.size), dtype=np.complex128)
        he2[0, ...] = Je[1, ...]*he1[2, ...] - Je[2, ...]*he1[1, ...]
        he2[1, ...] = Je[2, ...]*he1[0, ...] - Je[0, ...]*he1[2, ...]
        he2[2, ...] = Je[0, ...]*he1[1, ...] - Je[1, ...]*he1[0, ...]
        He = np.sum(he*he2*N*ds, axis=1)
        
        # Step 6: Store field components with normalization factors
        # E-field normalization: Z₀/(4π)
        Field_E_x[i] = Z0/(4*np.pi) * Ee[0]
        Field_E_y[i] = Z0/(4*np.pi) * Ee[1]
        Field_E_z[i] = Z0/(4*np.pi) * Ee[2]
        
        # H-field normalization: 1/(4π) [from Maxwell's equations]
        Field_H_x[i] = 1/(4*np.pi) * He[0]
        Field_H_y[i] = 1/(4*np.pi) * He[1]
        Field_H_z[i] = 1/(4*np.pi) * He[2]
    
    return Field_E_x, Field_E_y, Field_E_z, Field_H_x, Field_H_y, Field_H_z


@njit(parallel=True)
def _po_integrate_surface_currents_jit(x1, y1, z1, x2, y2, z2, N, ds, Je, Jm, k, Z0):
    """
    Compute EM fields from surface electric AND magnetic current densities.
    
    This is the general form of the Kirchhoff-Huygens integral, accounting
    for both electric (Je) and magnetic (Jm) surface current distributions.
    This occurs for apertures in conducting screens or for general scatterer
    surface current formulations.
    
    Mathematical Formulation:
    -------------------------
    Total electric field:
    
        E = Z₀/(4π) ∫ G(k) [k² Je(1j/kr - 1/kr² + 1j/kr³) + ...] ds'
          - 1/(4π) ∫ G(k) [k² Jm × R̂ (1 - 1j·kr)/(kr²)] ds'
    
    Total magnetic field:
    
        H = 1/(4π) ∫ G(k) [k² Je × R̂ (1 - 1j·kr)/(kr²)] ds'
          + 1/(4π·Z₀) ∫ G(k) [k² Jm(1j/kr - 1/kr² + 1j/kr³) + ...] ds'
    
    Where magnetic current Jm appears with dual role (analogous to Je but
    with permittivity-permeability duality).
    
    Parameters:
    -----------
    x1, y1, z1 : ndarray
        Source surface coordinates (shape: (N_source,))
    x2, y2, z2 : ndarray
        Observation surface coordinates (shape: (N_obs,))
    N : ndarray
        Jacobian factor for surface projection to 2D integration plane
        (accounts for surface orientation and projection geometry)
        Shape: (N_source,)
    ds : ndarray
        Differential area × sampling quadrature weights
        (combines element area with quadrature weights e.g., from Gaussian quadrature)
        Shape: (N_source,)
    Je : ndarray
        Electric surface current density
        Shape: (3, N_source) containing [Jex, Jey, Jez] components
    Jm : ndarray
        Magnetic surface current density (dual to Je)
        Shape: (3, N_source) containing [Jmx, Jmy, Jmz] components
    k : float
        Wave number (k = 2π/λ)
    Z0 : float
        Impedance of free space (~377 Ohm)
    
    Returns:
    --------
    Field_E_x, Field_E_y, Field_E_z : ndarray
        Electric field components at observation points
        Shape: (N_obs,), dtype: complex128
    Field_H_x, Field_H_y, Field_H_z : ndarray
        Magnetic field components at observation points
        Shape: (N_obs,), dtype: complex128
    
    Notes:
    ------
    1. This function is JIT-compiled with Numba for maximum performance
    2. Parallel loop over observation points
    3. Includes terms from both Je and Jm contributions
    4. Duality: Jm coupling uses 1/Z₀ factor instead of Z₀
    5. Cross products computed explicitly (no vector library dependency)
    
    Algorithm Steps:
    1. For each observation point i:
        a. Compute R, r, kr as in _po_integrate_surface_currents_je
        b. Compute E-field from Je term using vector potential
        c. Compute E-field from Jm term (dual coupling)
        d. Compute H-field from Je term (cross product)
        e. Compute H-field from Jm term using vector potential
        f. Sum contributions and apply amplitude factors
    
    Physical Interpretation:
    ------------------------
    - Je term: Electric currents radiate E-field directly
    - Jm term: Magnetic currents radiate H-field directly
    - Cross coupling: Je→H (via curl), Jm→E (via dual form)
    
    Example:
    --------
    >>> # Slot antenna: magnetic current in slot
    >>> x1 = np.linspace(-0.5, 0.5, 101)
    >>> y1 = np.zeros_like(x1)
    >>> z1 = np.zeros_like(x1)
    >>> x2 = np.array([0.0])
    >>> y2 = np.array([0.1])
    >>> z2 = np.array([0.0])
    >>> N = np.ones(101)
    >>> ds = np.ones(101) * 0.01
    >>> Je = np.zeros((3, 101))  # No electric current
    >>> Jm = np.vstack([np.zeros(101), np.ones(101), np.zeros(101)])  # Slot current
    >>> k = 2 * np.pi
    >>> Z0 = 377.0
    >>> Ex, Ey, Ez, Hx, Hy, Hz = _po_integrate_surface_currents(
    ...     x1, y1, z1, x2, y2, z2, N, ds, Je, Jm, k, Z0)
    """
    
    Field_E_x = np.zeros(x2.shape, dtype=np.complex128)
    Field_E_y = np.zeros(x2.shape, dtype=np.complex128)
    Field_E_z = np.zeros(x2.shape, dtype=np.complex128)
    Field_H_x = np.zeros(x2.shape, dtype=np.complex128)
    Field_H_y = np.zeros(x2.shape, dtype=np.complex128)
    Field_H_z = np.zeros(x2.shape, dtype=np.complex128)
    
    # Iterate over all observation points (parallel)
    for i in prange(x2.size):
        # Step 1: Compute relative position vector
        R = np.zeros((3, x1.size))
        R[0, ...] = x2[i] - x1.reshape(1, -1)
        R[1, ...] = y2[i] - y1.reshape(1, -1)
        R[2, ...] = z2[i] - z1.reshape(1, -1)
        
        # Step 2: Compute distances
        r = np.sqrt(np.sum(R**2, axis=0))
        kr = k * r  # kr = k·r (normalized distance)
        kr2 = kr**2  # (kr)²
        kr3 = kr**3  # (kr)³
        kr2_inv = 1.0 / kr2  # 1/(kr²)
        kr3_inv = 1.0 / kr3  # 1/(kr³)
        
        # Step 3: Compute Green's function phase
        phase = -kr
        
        # ===== ELECTRIC FIELD COMPUTATION =====
        
        # E-field contribution from electric current Je
        ee = np.exp(1j*phase) * k**2 * (
            Je * (1j/kr - kr2_inv + 1j*kr3_inv) +
            np.sum(Je*R/r, axis=0) * R/r * (-1j/kr + 3*kr2_inv - 3j*kr3_inv)
        )
        Ee = np.sum(ee * N * ds, axis=1)
        
        # E-field contribution from magnetic current Jm (cross coupling)
        # Dual form: (1/Z₀) Jm × R̂ (1 - jkr)/kr²
        em = np.exp(1j*phase) * k**2
        em1 = R/r * kr2_inv * (1 - 1j*kr)  # R̂ × (1 - j·kr)/(kr²)
        
        em2 = np.zeros((3, x1.size), dtype=np.complex128)
        em2[0, ...] = Jm[1, ...]*em1[2, ...] - Jm[2, ...]*em1[1, ...]
        em2[1, ...] = Jm[2, ...]*em1[0, ...] - Jm[0, ...]*em1[2, ...]
        em2[2, ...] = Jm[0, ...]*em1[1, ...] - Jm[1, ...]*em1[0, ...]
        Em = np.sum(em*em2*N*ds, axis=1)
        
        # ===== MAGNETIC FIELD COMPUTATION =====
        
        # H-field contribution from electric current Je (cross coupling)
        he = np.exp(1j*phase) * k**2
        he1 = R/r * kr2_inv * (1 - 1j*kr)  # R̂ × (1 - j·kr)/(kr²)
        
        he2 = np.zeros((3, x1.size), dtype=np.complex128)
        he2[0, ...] = Je[1, ...]*he1[2, ...] - Je[2, ...]*he1[1, ...]
        he2[1, ...] = Je[2, ...]*he1[0, ...] - Je[0, ...]*he1[2, ...]
        he2[2, ...] = Je[0, ...]*he1[1, ...] - Je[1, ...]*he1[0, ...]
        He = np.sum(he*he2*N*ds, axis=1)
        
        # H-field contribution from magnetic current Jm
        hm = np.exp(1j*phase) * k**2 * (
            Jm * (1j/kr - kr2_inv + 1j*kr3_inv) +
            np.sum(Jm*R/r, axis=0) * R/r * (-1j/kr + 3*kr2_inv - 3j*kr3_inv)
        )
        Hm = np.sum(hm*N*ds, axis=1)
        
        # Step 6: Combine contributions with proper scaling
        # E-field = Z₀/(4π)·Ee - 1/(4π)·Em
        Field_E_x[i] = Z0/(4*np.pi)*Ee[0] - 1/(4*np.pi)*Em[0]
        Field_E_y[i] = Z0/(4*np.pi)*Ee[1] - 1/(4*np.pi)*Em[1]
        Field_E_z[i] = Z0/(4*np.pi)*Ee[2] - 1/(4*np.pi)*Em[2]
        
        # H-field = 1/(4π)·He + 1/(4π·Z₀)·Hm
        Field_H_x[i] = 1/(4*np.pi)*He[0] + 1/(4*np.pi*Z0)*Hm[0]
        Field_H_y[i] = 1/(4*np.pi)*He[1] + 1/(4*np.pi*Z0)*Hm[1]
        Field_H_z[i] = 1/(4*np.pi)*He[2] + 1/(4*np.pi*Z0)*Hm[2]
    
    return Field_E_x, Field_E_y, Field_E_z, Field_H_x, Field_H_y, Field_H_z
