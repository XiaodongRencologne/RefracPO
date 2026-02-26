#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import time
import copy

import numpy as np
import torch as T
from torch.cuda.amp import autocast, GradScaler
cpu_cores = T.get_num_threads()
print(cpu_cores)
T.set_num_threads(cpu_cores*2)

from numba import njit, prange
from tqdm import tqdm

from .vecops import dot, cross, Vector as vector
from . import c, mu, epsilon, Z0
from .po_core_gpu import po_integrate_surface_currents_gpu, _process_FFcal_batch_gpu

def PO_GPU_2(src_face, src_face_n, src_face_dS,
             obs_face,
             Field_in_E, Field_in_H,
             k, n,
             device=T.device('cuda')):
    """
    Optimized Physical Optics (PO) GPU implementation for near-field calculations.

    This function computes electromagnetic fields on the observation surface due to
    incident fields on the source surface using the physical optics approximation and
    GPU acceleration.
    
    Physical Model:
        - Converts incident fields to surface currents using PO boundary conditions
        - Integrates Kirchhoff-Huygens integrals to compute radiated fields
        - Supports both electric (Je) and magnetic (Jm) currents
        - Accounts for material properties through refractive index n

    Parameters:
        src_face: Surface geometry object with .x, .y, .z arrays (source surface)
        src_face_n: Surface normal vector object with .N array (normal amplitude weighting)
        src_face_dS: Differential area elements on source surface (shape: N_src)
        obs_face: Surface geometry object with .x, .y, .z arrays (observation surface)
        Field_in_E: Incident electric field (Vector object)
        Field_in_H: Incident magnetic field (Vector object)
        k: Wave number in free space (real number)
        n: Refractive index of material on source surface side
        device: PyTorch device for computation (default: T.device('cuda'))

    Returns:
        (Field_E, Field_H): Computed electric and magnetic fields on observation surface
            - Field_E: Vector object with field components (complex128)
            - Field_H: Vector object with field components (complex128)
            
    Physical Interpretation:
        Field_E: Electric field resulting from surface currents (includes both Je and Jm)
        Field_H: Magnetic field resulting from surface currents (includes both Je and Jm)

    Notes:
        - Automatically converts between Vector objects and PyTorch tensors
        - Handles batch processing for large observation surfaces
        - Uses mixed precision (autocast) for efficiency
        - Returns results as Vector objects in NumPy format (CPU)
    """
    
    # Validate inputs
    if device is None:
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    
    # Convert to PyTorch device if string
    if isinstance(device, str):
        device = T.device(device)
    
    # Physical constants and parameters
    k_n = k * n  # Wave number in material
    
    # Compute weighting factors for surface elements
    # N = normal magnitude × area scaling / (4π)
    # ds = element area × normal weight × (k_n)^2 / (4π)
    N = src_face_n.N
    ds = src_face_dS * N * (k_n ** 2) / (4 * np.pi)
    
    # Compute equivalent currents with source-area weighting
    # Je = 2 * ds * (n_hat × H), Jm = 2 * ds * (n_hat × E)
    Je = 2 * ds * cross(src_face_n, Field_in_H)
    Jm = 2 * ds * cross(src_face_n, Field_in_E)
    
    # Convert Vector objects to PyTorch tensors with shape (3, N_src)
    r_src = T.stack([
        T.tensor(src_face.x.ravel(), dtype=T.float64, device=device),
        T.tensor(src_face.y.ravel(), dtype=T.float64, device=device),
        T.tensor(src_face.z.ravel(), dtype=T.float64, device=device)
    ], dim=0)
    
    r_obs = T.stack([
        T.tensor(obs_face.x.ravel(), dtype=T.float64, device=device),
        T.tensor(obs_face.y.ravel(), dtype=T.float64, device=device),
        T.tensor(obs_face.z.ravel(), dtype=T.float64, device=device)
    ], dim=0)
    
    # Convert surface currents to tensors
    Je_tensor = T.stack([
        T.tensor(Je.x, dtype=T.complex128, device=device),
        T.tensor(Je.y, dtype=T.complex128, device=device),
        T.tensor(Je.z, dtype=T.complex128, device=device)
    ], dim=0)
    
    Jm_tensor = T.stack([
        T.tensor(Jm.x, dtype=T.complex128, device=device),
        T.tensor(Jm.y, dtype=T.complex128, device=device),
        T.tensor(Jm.z, dtype=T.complex128, device=device)
    ], dim=0)
    
    # Convert weighting factors to tensors
    ds_tensor = T.tensor(ds, dtype=T.float64, device=device)
    
    # Call GPU core function
    # This function handles batch processing internally
    Field_E_tensor, Field_H_tensor = po_integrate_surface_currents_gpu(
        r_src=r_src,
        r_obs=r_obs,
        ds=ds_tensor,
        Je=Je_tensor,
        Jm=Jm_tensor,
        k=k_n,
        Z0_val=Z0,
        device=device
    )
    
    # Convert output tensors back to Vector objects (NumPy format)
    Field_E = vector()
    Field_E.x = Field_E_tensor[0].cpu().detach().numpy()
    Field_E.y = Field_E_tensor[1].cpu().detach().numpy()
    Field_E.z = Field_E_tensor[2].cpu().detach().numpy()
    
    Field_H = vector()
    Field_H.x = Field_H_tensor[0].cpu().detach().numpy()
    Field_H.y = Field_H_tensor[1].cpu().detach().numpy()
    Field_H.z = Field_H_tensor[2].cpu().detach().numpy()
    
    # Clean up GPU memory
    T.cuda.empty_cache()
    T.cuda.synchronize()
    
    return Field_E, Field_H


# --- Backward Compatibility Layer ---
# These functions provide support for legacy code using old Vector operation names
def PO_far_GPU2(src_face,src_face_n,src_face_dS,
               obs_face,
               Field_in_E,
               Field_in_H,
               k,
               device =T.device('cuda')):
    """
    Optimized Physical Optics (PO) GPU implementation for far-field calculations.

    This function computes far-field electromagnetic radiation on observation points
    (obs_face) from equivalent surface currents induced on source surface (src_face).
    It uses a plane-wave phase model and GPU batching for memory-efficient execution.

    Physical Model:
        - Builds equivalent electric current from incident magnetic field
        - Builds equivalent magnetic current from incident electric field
        - Applies far-field phase term exp(j k r_hat · r_src)
        - Computes radiated E/H fields via cross-product radiation operators

    Parameters:
        src_face: Source surface geometry object with .x, .y, .z arrays
        src_face_n: Surface normal vector object with .N array (normal weighting)
        src_face_dS: Differential area elements on src_face (shape: N_src)
        obs_face: Observation direction/point geometry object with .x, .y, .z arrays
        Field_in_E: Incident electric field on src_face (Vector object)
        Field_in_H: Incident magnetic field on src_face (Vector object)
        k: Wave number in the current medium
        device: PyTorch device for computation (default: T.device('cuda'))

    Returns:
        (Field_E, Field_H): Far-field electric and magnetic fields on obs_face
            - Field_E: Vector object with complex field components
            - Field_H: Vector object with complex field components

    Notes:
        - Keeps tensors on device during batch accumulation
        - Batch size is estimated from available GPU memory
        - Converts outputs to NumPy arrays once at the end
    """

    # Validate inputs
    if device is None:
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    
    # Convert to PyTorch device if string
    if isinstance(device, str):
        device = T.device(device)
    
    # Compute weighting factors for surface elements
    # N = normal magnitude × area scaling / (4π)
    # ds = element area × normal weight × (k_n)^2 / (4π)
    N = src_face_n.N
    ds = src_face_dS * N * (k ** 2) / (4 * np.pi)
    
    # Compute equivalent currents with source-element weighting
    # Je = 2 * ds * (n_hat × H), Jm = 2 * ds * (n_hat × E)
    Je = 2 * ds * cross(src_face_n, Field_in_H)
    Jm = 2 * ds * cross(src_face_n, Field_in_E)
    
    # Convert Vector objects to PyTorch tensors with shape (3, N_src)
    r_src = T.stack([
        T.tensor(src_face.x.ravel(), dtype=T.float64, device=device),
        T.tensor(src_face.y.ravel(), dtype=T.float64, device=device),
        T.tensor(src_face.z.ravel(), dtype=T.float64, device=device)
    ], dim=0)
    
    r_obs = T.stack([
        T.tensor(obs_face.x.ravel(), dtype=T.float64, device=device),
        T.tensor(obs_face.y.ravel(), dtype=T.float64, device=device),
        T.tensor(obs_face.z.ravel(), dtype=T.float64, device=device)
    ], dim=0)
    
    # Convert surface currents to tensors
    Je_tensor = T.stack([
        T.tensor(Je.x, dtype=T.complex128, device=device),
        T.tensor(Je.y, dtype=T.complex128, device=device),
        T.tensor(Je.z, dtype=T.complex128, device=device)
    ], dim=0)
    
    Jm_tensor = T.stack([
        T.tensor(Jm.x, dtype=T.complex128, device=device),
        T.tensor(Jm.y, dtype=T.complex128, device=device),
        T.tensor(Jm.z, dtype=T.complex128, device=device)
    ], dim=0)

    # Convert weighting factors to tensors
    ds_tensor = T.tensor(ds, dtype=T.float64, device=device)
    
    # Call GPU core function
    # This function handles batch processing internally
    Field_E_tensor, Field_H_tensor = po_integrate_surface_currents_gpu(
        r_src=r_src,
        r_obs=r_obs,
        ds=ds_tensor,
        Je=Je_tensor,
        Jm=Jm_tensor,
        k=k,
        far_field=True,
        Z0_val=Z0,
        device=device
    )
    
    # Convert output tensors back to Vector objects (NumPy format)
    Field_E = vector()
    Field_E.x = Field_E_tensor[0].cpu().detach().numpy()
    Field_E.y = Field_E_tensor[1].cpu().detach().numpy()
    Field_E.z = Field_E_tensor[2].cpu().detach().numpy()
    
    Field_H = vector()
    Field_H.x = Field_H_tensor[0].cpu().detach().numpy()
    Field_H.y = Field_H_tensor[1].cpu().detach().numpy()
    Field_H.z = Field_H_tensor[2].cpu().detach().numpy()
    
    # Clean up GPU memory
    T.cuda.empty_cache()
    T.cuda.synchronize()
    
    return Field_E, Field_H