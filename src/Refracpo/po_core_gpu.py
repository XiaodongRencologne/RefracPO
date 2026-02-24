"""
Physical Optics (PO) Core GPU Calculation Module
==================================================

This module contains GPU-optimized electromagnetic field calculations
for Physical Optics analysis using PyTorch tensors.

Key Features:
    - PyTorch-based tensor operations (CPU and GPU compatible)
    - **Adaptive batch processing with memory-efficient computation**
    - Support for complex field calculations
    - Device agnostic (CPU/GPU)
    - **Automatic GPU memory monitoring and batching**

Functions:
    po_integrate_surface_currents_je_gpu: Integrate E-field only (Jm=0) on GPU
    po_integrate_surface_currents_gpu: Integrate E-field and M-field (Jm≠0) on GPU
    
Notes:
    - All inputs and outputs use PyTorch tensors
    - Supports mixed precision (autocast) for faster computation
    - **Memory management includes adaptive batching for out-of-memory prevention**
    - **Batch size automatically adjusted based on available GPU memory**
"""

import torch as T
import numpy as np
from .RefracPO.src.Refracpo import c, mu, epsilon, Z0


def _estimate_required_memory(N_obs, N_src, device, complex_fields=False):
    """
    Estimate required GPU memory for a computation.
    
    Parameters:
        N_obs: Number of observation points
        N_src: Number of source points
        device: torch device
        complex_fields: Whether to include both Je and Jm (doubles memory)
    
    Returns:
        Memory in bytes
    """
    # Main intermediate tensor: (3, N_obs, N_src) with complex128
    # complex128 = 16 bytes per element
    base_memory = 3 * N_obs * N_src * 16  # bytes
    
    if complex_fields:
        # Additional tensors for Jm computation
        base_memory *= 2
    
    # Add overhead for intermediate computations (30% safety margin)
    return int(base_memory * 1.3)


def _get_available_memory(device):
    """
    Get available GPU/CPU memory.
    
    Returns:
        Available memory in bytes
    """
    if device.type == 'cuda':
        # GPU memory management
        total_memory = T.cuda.get_device_properties(0).total_memory
        allocated_memory = T.cuda.memory_allocated(0)
        reserved_memory = T.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        return max(0, available_memory)
    else:
        # For CPU, assume we can use a reasonable amount
        # In practice, this is harder to estimate accurately
        # Use a conservative estimate (512 MB for safety)
        return 512 * 1024 * 1024


def _calculate_optimal_batch_size(N_obs, N_src, device, complex_fields=False, 
                                   memory_fraction=0.8, min_batch=1):
    """
    Calculate optimal batch size (N_obs batch) for memory-efficient computation.
    
    Parameters:
        N_obs: Total number of observation points
        N_src: Number of source points
        device: torch device
        complex_fields: Whether computation includes both Je and Jm
        memory_fraction: Fraction of available memory to use (0.0-1.0)
        min_batch: Minimum batch size (at least this many obs points per batch)
    
    Returns:
        Batch size (number of observation points per batch)
    """
    available_memory = _get_available_memory(device)
    
    # We need memory for: (3, batch_size, N_src) complex tensor
    # complex128 = 16 bytes
    bytes_per_obs = 3 * N_src * 16  # Base tensor
    
    if complex_fields:
        bytes_per_obs *= 2  # Extra computation for Jm
    
    # Add 30% overhead for intermediate computations
    bytes_per_obs = int(bytes_per_obs * 1.3)
    
    # Calculate maximum batch size based on available memory
    usable_memory = int(available_memory * memory_fraction)
    max_batch_size = max(min_batch, usable_memory // bytes_per_obs)
    
    # Ensure we don't exceed total observation points
    batch_size = min(max_batch_size, N_obs)
    
    return max(min_batch, batch_size)


def _process_batch_je_gpu(r_src, r_obs_batch, N, ds, Je, k, Z0_tensor, device):
    """
    Process a single batch of observation points for Je-only computation.
    
    Returns:
        (Field_E_batch, Field_H_batch) for this batch
    """
    N_obs_batch = r_obs_batch.shape[1]
    N_src = r_src.shape[1]
    
    # Compute relative position vectors: R = r_obs - r_src
    R = r_obs_batch.unsqueeze(2) - r_src.unsqueeze(1)  # (3, N_obs_batch, N_src)
    
    # Compute distances
    r = T.linalg.norm(R, dim=0)  # (N_obs_batch, N_src)
    kr = k * r
    kr2 = kr ** 2
    kr3 = kr ** 3
    kr2_inv = 1.0 / kr2
    kr3_inv = 1.0 / kr3
    
    # Normalize R
    R_normalized = R / r.unsqueeze(0)  # (3, N_obs_batch, N_src)
    
    # Green's function phase
    phase = -kr
    green_func = T.exp(1j * phase)  # (N_obs_batch, N_src)
    
    # Electric field computation
    Je_dot_R = T.sum(Je.unsqueeze(1) * R_normalized, dim=0)  # (N_obs_batch, N_src)
    
    ee1 = Je.unsqueeze(1) * (1j/kr - kr2_inv + 1j*kr3_inv)
    ee2 = R_normalized * (Je_dot_R * (-1j/kr + 3*kr2_inv - 3j*kr3_inv)).unsqueeze(0)
    
    ee = green_func.unsqueeze(0) * k**2 * (ee1 + ee2)  # (3, N_obs_batch, N_src)
    
    # Apply weighting N * ds
    weighted_ee = ee *  ds.unsqueeze(0).unsqueeze(0)
    Ee = T.sum(weighted_ee, dim=2)  # (3, N_obs_batch)
    
    # Magnetic field computation via cross product
    he_factor = R_normalized * kr2_inv * (1.0 - 1j*kr)
    he = T.cross(Je.unsqueeze(1).expand(-1, N_obs_batch, -1), 
                 he_factor, dim=0)  # (3, N_obs_batch, N_src)
    
    he_weighted = he * green_func.unsqueeze(0) * k**2 * (N * ds).unsqueeze(0).unsqueeze(0)
    He = T.sum(he_weighted, dim=2)  # (3, N_obs_batch)
    
    # Apply normalization factors
    Field_E_batch = Z0_tensor / (4 * np.pi) * Ee
    Field_H_batch = 1.0 / (4 * np.pi) * He
    
    return Field_E_batch, Field_H_batch


def _process_batch_gpu(r_src, r_obs_batch, N, ds, Je, Jm, k, Z0_tensor, device):
    """
    Process a single batch of observation points for Je+Jm computation.
    
    Returns:
        (Field_E_batch, Field_H_batch) for this batch
    """
    N_obs_batch = r_obs_batch.shape[1]
    N_src = r_src.shape[1]
    
    # Compute relative position vectors
    R = r_obs_batch.unsqueeze(2) - r_src.unsqueeze(1)  # (3, N_obs_batch, N_src)
    
    # Compute distances
    r = T.linalg.norm(R, dim=0)  # (N_obs_batch, N_src)
    kr = k * r
    kr2 = kr ** 2
    kr3 = kr ** 3
    kr2_inv = 1.0 / kr2
    kr3_inv = 1.0 / kr3
    
    # Normalize R
    R_normalized = R / r.unsqueeze(0)  # (3, N_obs_batch, N_src)
    
    # Green's function phase
    phase = -kr
    green_func = T.exp(1j * phase)  # (N_obs_batch, N_src)
    
    # Weighting factor
    weight = ds.unsqueeze(0).unsqueeze(0)  # (1, 1, N_src)
    
    # ===== ELECTRIC FIELD COMPUTATION =====
    
    # E-field from Je
    Je_dot_R = T.sum(Je.unsqueeze(1) * R_normalized, dim=0)
    
    ee1 = Je.unsqueeze(1) * (1j/kr - kr2_inv + 1j*kr3_inv)
    ee2 = R_normalized * (Je_dot_R * (-1j/kr + 3*kr2_inv - 3j*kr3_inv)).unsqueeze(0)
    
    ee = green_func.unsqueeze(0) * k**2 * (ee1 + ee2)
    Ee = T.sum(ee * weight, dim=2)
    
    # E-field from Jm (cross coupling)
    he_factor = R_normalized * kr2_inv * (1.0 - 1j*kr)
    em = T.cross(Jm.unsqueeze(1).expand(-1, N_obs_batch, -1), 
                 he_factor, dim=0)
    
    em_weighted = em * green_func.unsqueeze(0) * k**2 * weight
    Em = T.sum(em_weighted, dim=2)
    
    # ===== MAGNETIC FIELD COMPUTATION =====
    
    # H-field from Je (cross coupling)
    he = T.cross(Je.unsqueeze(1).expand(-1, N_obs_batch, -1), 
                 he_factor, dim=0)
    
    he_weighted = he * green_func.unsqueeze(0) * k**2 * weight
    He = T.sum(he_weighted, dim=2)
    
    # H-field from Jm
    Jm_dot_R = T.sum(Jm.unsqueeze(1) * R_normalized, dim=0)
    
    hm1 = Jm.unsqueeze(1) * (1j/kr - kr2_inv + 1j*kr3_inv)
    hm2 = R_normalized * (Jm_dot_R * (-1j/kr + 3*kr2_inv - 3j*kr3_inv)).unsqueeze(0)
    
    hm = green_func.unsqueeze(0) * k**2 * (hm1 + hm2)
    Hm = T.sum(hm * weight, dim=2)
    
    # Combine contributions with proper scaling
    Field_E_batch = Z0_tensor / (4 * np.pi) * Ee - 1.0 / (4 * np.pi) * Em
    Field_H_batch = 1.0 / (4 * np.pi) * He + 1.0 / (4 * np.pi * Z0_tensor) * Hm
    
    return Field_E_batch, Field_H_batch


def po_integrate_surface_currents_je_gpu(r_src, r_obs, N, ds, Je, k, Z0_val=None, device=None, 
                                          max_batch_size=None, memory_fraction=0.8):
    """
    Compute EM fields from surface electric current density (Jm = 0) using GPU computation.
    
    **Includes memory-efficient adaptive batching to handle large problems.**
    
    This is the GPU-optimized version using PyTorch tensors with batch processing.
    
    Mathematical Formulation:
    -------------------------
    The electric field is computed from the electric current distribution:
    
        E = ∫ G(k) · [k² Je(1j/kr - 1/kr² + 1j/kr³) 
                      + (Je·R̂) R̂ (-1j/kr + 3/kr² - 3j/kr³)]
            × N(r') ds' dω
    
    The magnetic field is:
    
        H = (1/Zf) ∫ k² G(k) R̂ × Je (1 - 1j·kr)/(kr²) × N(r') ds' dω
    
    Parameters:
    -----------
    r_src : torch.Tensor
        Source surface position vectors (3, N_source)
        Shape: (3, N_source)
    r_obs : torch.Tensor
        Observation surface position vectors (3, N_obs)
        Shape: (3, N_obs)
    N : torch.Tensor
        Jacobian factor for surface projection to 2D integration plane
        (accounts for surface orientation and projection geometry)
        Shape: (N_source,)
    ds : torch.Tensor
        Differential area × sampling quadrature weights
        (combines element area with quadrature weights e.g., from Gaussian quadrature)
        Shape: (N_source,)
    Je : torch.Tensor
        Electric surface current density
        Shape: (3, N_source) containing [Jex, Jey, Jez] components
    k : float or torch.Tensor
        Wave number (k = 2π/λ)
    Z0_val : float, optional
        Impedance of free space (default: module constant)
    device : torch.device, optional
        Device to use for computation (default: auto-detect from tensors)
    max_batch_size : int, optional
        Maximum batch size for observation points (default: auto-calculate from memory)
    memory_fraction : float, optional
        Fraction of available memory to use (0.0-1.0, default: 0.8)
    
    Returns:
    --------
    Field_E : torch.Tensor
        Electric field at observation points (3, N_obs)
    Field_H : torch.Tensor
        Magnetic field at observation points (3, N_obs)
    
    Algorithm:
    ----------
    For each observation point i:
        1. Compute vector R from all source points to observation point
        2. Compute distance r = |R|
        3. Compute normalized distance kr = k·r and its powers
        4. Compute Green's function phase: exp(-jkr)
        5. Compute electric field contribution
        6. Compute magnetic field contribution via cross product
        7. Sum contributions from all source points with weighting N·ds
        8. Apply normalization factors Z0/(4π) and 1/(4π)
        
    Memory Management:
    ------------------
    - Automatically detects available GPU/CPU memory
    - Adaptively adjusts batch size to prevent out-of-memory errors
    - Processes observation points in batches if needed
    - Cleans up intermediate tensors to minimize memory usage
    """
    
    if Z0_val is None:
        Z0_val = Z0
    
    if device is None:
        device = r_obs.device
    
    # Ensure all inputs are on the same device
    r_src = r_src.to(device)
    r_obs = r_obs.to(device)
    N = N.to(device)
    ds = ds.to(device)
    Je = Je.to(device)
    
    if isinstance(k, (int, float)):
        k = T.tensor(k, dtype=T.float64, device=device)
    else:
        k = k.to(device)
    
    Z0_tensor = T.tensor(Z0_val, dtype=T.complex128, device=device)
    
    # Initialize output tensors
    N_obs = r_obs.shape[1]
    N_src = r_src.shape[1]
    
    # Calculate optimal batch size if not provided
    if max_batch_size is None:
        max_batch_size = _calculate_optimal_batch_size(
            N_obs, N_src, device, 
            complex_fields=False, 
            memory_fraction=memory_fraction,
            min_batch=1
        )
    
    # If batch size is larger than N_obs, process all at once
    if max_batch_size >= N_obs:
        # Process all observation points at once
        return _process_batch_je_gpu(r_src, r_obs, N, ds, Je, k, Z0_tensor, device)
    
    # Otherwise, process in batches
    Field_E = T.zeros((3, N_obs), dtype=T.complex128, device=device)
    Field_H = T.zeros((3, N_obs), dtype=T.complex128, device=device)
    
    num_batches = (N_obs + max_batch_size - 1) // max_batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_batch_size
        end_idx = min((batch_idx + 1) * max_batch_size, N_obs)
        
        # Extract batch of observation points
        r_obs_batch = r_obs[:, start_idx:end_idx]
        
        # Process batch
        Field_E_batch, Field_H_batch = _process_batch_je_gpu(
            r_src, r_obs_batch, N, ds, Je, k, Z0_tensor, device
        )
        
        # Store results
        Field_E[:, start_idx:end_idx] = Field_E_batch
        Field_H[:, start_idx:end_idx] = Field_H_batch
        
        # Clean up GPU cache periodically
        if device.type == 'cuda' and batch_idx % 10 == 9:
            T.cuda.empty_cache()
    
    return Field_E, Field_H


def po_integrate_surface_currents_gpu(r_src, r_obs, ds, 
                                      Je, Jm, k, 
                                      Z0_val=None, device=None,
                                       max_batch_size=None, memory_fraction=0.8):
    """
    Compute EM fields from surface electric AND magnetic current densities using GPU computation.
    
    **Includes memory-efficient adaptive batching to handle large problems.**
    
    This is the GPU-optimized version using PyTorch tensors with batch processing.
    Implements the general Kirchhoff-Huygens integral for both Je and Jm.
    
    Mathematical Formulation:
    -------------------------
    Total electric field:
    
        E = Z₀/(4π) ∫ G(k) [k² Je(...) + ...] ds'
          - 1/(4π) ∫ G(k) [k² Jm × R̂ (1 - 1j·kr)/(kr²)] ds'
    
    Total magnetic field:
    
        H = 1/(4π) ∫ G(k) [k² Je × R̂ (1 - 1j·kr)/(kr²)] ds'
          + 1/(4π·Z₀) ∫ G(k) [k² Jm(...) + ...] ds'
    
    Parameters:
    -----------
    r_src : torch.Tensor
        Source surface position vectors (3, N_source)
    r_obs : torch.Tensor
        Observation surface position vectors (3, N_obs)
    N : torch.Tensor
        Jacobian factor for surface projection to 2D integration plane
        (accounts for surface orientation and projection geometry)
        Shape: (N_source,)
    ds : torch.Tensor
        Differential area × sampling quadrature weights
        (combines element area with quadrature weights e.g., from Gaussian quadrature)
        Shape: (N_source,)
    Je : torch.Tensor
        Electric surface current density (shape: (3, N_source))
    Jm : torch.Tensor
        Magnetic surface current density (shape: (3, N_source))
    k : float or torch.Tensor
        Wave number (k = 2π/λ)
    Z0_val : float, optional
        Impedance of free space (default: module constant)
    device : torch.device, optional
        Device to use for computation (default: auto-detect from tensors)
    max_batch_size : int, optional
        Maximum batch size for observation points (default: auto-calculate from memory)
    memory_fraction : float, optional
        Fraction of available memory to use (0.0-1.0, default: 0.8)
    
    Returns:
    --------
    Field_E : torch.Tensor
        Electric field at observation points (3, N_obs)
    Field_H : torch.Tensor
        Magnetic field at observation points (3, N_obs)
        
    Memory Management:
    ------------------
    - Automatically detects available GPU/CPU memory
    - Adaptively adjusts batch size to prevent out-of-memory errors
    - Processes observation points in batches if needed
    - Cleans up intermediate tensors to minimize memory usage
    """
    
    if Z0_val is None:
        Z0_val = Z0
    
    if device is None:
        device = r_obs.device
    
    # Ensure all inputs are on the same device
    r_src = r_src.to(device)
    r_obs = r_obs.to(device)
    ds = ds.to(device)
    Je = Je.to(device)
    Jm = Jm.to(device)
    
    if isinstance(k, (int, float)):
        k = T.tensor(k, dtype=T.float64, device=device)
    else:
        k = k.to(device)
    
    Z0_tensor = T.tensor(Z0_val, dtype=T.complex128, device=device)
    
    # Initialize output tensors
    N_obs = r_obs.shape[1]
    N_src = r_src.shape[1]
    
    # Calculate optimal batch size if not provided
    if max_batch_size is None:
        max_batch_size = _calculate_optimal_batch_size(
            N_obs, N_src, device,
            complex_fields=True,  # Both Je and Jm
            memory_fraction=memory_fraction,
            min_batch=1
        )
    
    # If batch size is larger than N_obs, process all at once
    if max_batch_size >= N_obs:
        # Process all observation points at once
        return _process_batch_gpu(r_src, r_obs, ds, Je, Jm, k, Z0_tensor, device)
    
    # Otherwise, process in batches
    Field_E = T.zeros((3, N_obs), dtype=T.complex128, device=device)
    Field_H = T.zeros((3, N_obs), dtype=T.complex128, device=device)
    
    num_batches = (N_obs + max_batch_size - 1) // max_batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_batch_size
        end_idx = min((batch_idx + 1) * max_batch_size, N_obs)
        
        # Extract batch of observation points
        r_obs_batch = r_obs[:, start_idx:end_idx]
        
        # Process batch
        Field_E_batch, Field_H_batch = _process_batch_gpu(
            r_src, r_obs_batch, ds, Je, Jm, k, Z0_tensor, device
        )
        
        # Store results
        Field_E[:, start_idx:end_idx] = Field_E_batch
        Field_H[:, start_idx:end_idx] = Field_H_batch
        
        # Clean up GPU cache periodically
        if device.type == 'cuda' and batch_idx % 10 == 9:
            T.cuda.empty_cache()
    
    return Field_E, Field_H
