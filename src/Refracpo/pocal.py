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
from ....po_core_gpu import po_integrate_surface_currents_gpu


def PO_cpu(src, src_n, src_ds, # geometry of source surface
           obs, # geometry of observation surface
           input_E, input_H, # Electromagnetic fields at source surface
           k, n, # wave number and refractive index
           ):
    pass

def PO(face1,face1_n,face1_dS,
    face2,
    Field_in_E,Field_in_H,
    k,parallel=True):
    # output field:
    Field_E=vector();
    Field_H=vector();    
    Je_in=scalarproduct(1,crossproduct(face1_n,Field_in_H));
    if Field_in_E==0:
        Jm_in=0;
    else:
        Jm_in=scalarproduct(1,crossproduct(face1_n,Field_in_E));
    JE=np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,-1);
    
    '''magnetic field is zero'''
    @njit(parallel=parallel)
    def calculus1(x1,y1,z1,x2,y2,z2,N,ds,Je): 
        Field_E_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        #R=np.zeros((3,x1.size));
        #he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
        for i in prange(x2.size): 
            R=np.zeros((3,x1.size));
            R[0,...]=x2[i]-x1.ravel();
            R[1,...]=y2[i]-y1.ravel();
            R[2,...]=z2[i]-z1.ravel();
            r=np.sqrt(np.sum(R**2,axis=0));
            
            '''calculate the vector potential Ae based on induced current'''
            phase=-k*r;
            r2=(k**2)*(r**2);
            r3=(k**3)*(r**3);
            '''1'''
            ee=np.exp(1j*phase)*k**2*(Je*(1j/phase-1/r2+1j/r3)+np.sum(Je*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
            Ee=np.sum(ee*N*ds,axis=1);
            '''2'''
            he=np.exp(1j*phase)*k**2
            he1=(R/r*1/(r2)*(1-1j*phase));
            he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
            he2[0,...]=Je[1,...]*he1[2,...]-Je[2,...]*he1[1,...];
            he2[1,...]=Je[2,...]*he1[0,...]-Je[0,...]*he1[2,...];
            he2[2,...]=Je[0,...]*he1[1,...]-Je[1,...]*he1[0,...];
            He=np.sum(he*he2*N*ds,axis=1);
            
            Field_E_x[i]=Z0/(4*np.pi)*Ee[0]
            Field_E_y[i]=Z0/(4*np.pi)*Ee[1]
            Field_E_z[i]=Z0/(4*np.pi)*Ee[2]
        
            Field_H_x[i]=1/4/np.pi*He[0]
            Field_H_y[i]=1/4/np.pi*He[1]
            Field_H_z[i]=1/4/np.pi*He[2]


        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z;
    '''Jm!=0'''
    @njit(parallel=parallel)
    def calculus2(x1,y1,z1,x2,y2,z2,N,ds,Je,Jm):
        Field_E_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        
        #em2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
        #he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
        for i in prange(x2.size):
            R=np.zeros((3,x1.size));
            R[0,...]=x2[i]-x1.reshape(1,-1);
            R[1,...]=y2[i]-y1.reshape(1,-1);
            R[2,...]=z2[i]-z1.reshape(1,-1);
            
            r=np.sqrt(np.sum(R**2,axis=0));
            
            
            '''calculate the vector potential Ae based on induced current'''
            phase=-k*r;
            r2=(k**2)*(r**2);
            r3=(k**3)*(r**3);
            '''1'''
            ee=np.exp(1j*phase)*k**2*(Je*(1j/phase-1/r2+1j/r3)+np.sum(Je*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
            Ee=np.sum(ee*N*ds,axis=1);
            '''2'''
            he=np.exp(1j*phase)*k**2
            he1=(R/r*1/(r2)*(1-1j*phase));
            he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
            he2[0,...]=Je[1,...]*he1[2,...]-Je[2,...]*he1[1,...];
            he2[1,...]=Je[2,...]*he1[0,...]-Je[0,...]*he1[2,...];
            he2[2,...]=Je[0,...]*he1[1,...]-Je[1,...]*he1[0,...];
            He=np.sum(he*he2*N*ds,axis=1);
            '''3'''
            em=np.exp(1j*phase)*k**2
            em1=(R/r*1/r2*(1-1j*phase));
            em2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
            em2[0,...]=Jm[1,...]*em1[2,...]-Jm[2,...]*em1[1,...];
            em2[1,...]=Jm[2,...]*em1[0,...]-Jm[0,...]*em1[2,...];
            em2[2,...]=Jm[0,...]*em1[1,...]-Jm[1,...]*em1[0,...];
            Em=np.sum(em*em2*N*ds,axis=1);
            '''4'''
            hm=np.exp(1j*phase)*k**2*(Jm*(1j/phase-1/r2+1j/r3)+np.sum(Jm*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
            Hm=np.sum(hm*N*ds,axis=1);
            
            Field_E_x[i]=Z0/(4*np.pi)*Ee[0]-1/(4*np.pi)*Em[0];
            Field_E_y[i]=Z0/(4*np.pi)*Ee[1]-1/(4*np.pi)*Em[1];
            Field_E_z[i]=Z0/(4*np.pi)*Ee[2]-1/(4*np.pi)*Em[2];
        
            Field_H_x[i]=1/4/np.pi*He[0]+1/(4*np.pi*Z0)*Hm[0];
            Field_H_y[i]=1/4/np.pi*He[1]+1/(4*np.pi*Z0)*Hm[1];
            Field_H_z[i]=1/4/np.pi*He[2]+1/(4*np.pi*Z0)*Hm[2];
            #Field_H_x[i]=1/Z0*Field_E_x[i]
            #Field_H_y[i]=1/Z0*Field_E_y[i]
            #Field_H_z[i]=1/Z0*Field_E_z[i]
            
            
            
        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z;
    if Jm_in==0:
        Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                                              face1_n.N,face1_dS,JE);
    else:
        JM=np.append(np.append(Jm_in.x,Jm_in.y),Jm_in.z).reshape(3,-1);
        Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus2(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                                              face1_n.N,face1_dS,JE,JM);
    return Field_E,Field_H;



def PO_GPU_2(face1, face1_n, face1_dS,
             face2,
             Field_in_E, Field_in_H,
             k, n,
             device=T.device('cuda')):
    """
    Optimized Physical Optics (PO) GPU implementation for near-field calculations.

    This function computes electromagnetic fields on surface 2 (face2) due to incident
    fields on surface 1 (face1) using the physical optics approximation and GPU acceleration.
    
    Physical Model:
        - Converts incident fields to surface currents using PO boundary conditions
        - Integrates Kirchhoff-Huygens integrals to compute radiated fields
        - Supports both electric (Je) and magnetic (Jm) currents
        - Accounts for material properties through refractive index n

    Parameters:
        face1: Surface geometry object with .x, .y, .z arrays (source surface)
        face1_n: Surface normal vector object with .N array (normal amplitude weighting)
        face1_dS: Differential area elements on face1 (shape: N_src)
        face2: Surface geometry object with .x, .y, .z arrays (observation surface)
        Field_in_E: Incident electric field (Vector object)
        Field_in_H: Incident magnetic field (Vector object)
        k: Wave number in free space (real number)
        n: Refractive index of material on face1
        device: PyTorch device for computation (default: T.device('cuda'))

    Returns:
        (Field_E, Field_H): Computed electric and magnetic fields on face2
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
    N = face1_n.N
    ds = face1_dS * N * (k_n ** 2) / (4 * np.pi)
    
    # Compute surface currents from incident fields using PO boundary conditions
    # Je = 2 * n_hat × H (electric current)
    # Jm = -2 * n_hat × E (magnetic current)
    Je = 2 * cross(face1_n, Field_in_H)  # Electric current
    Jm = 2 * cross(face1_n, Field_in_E)  # Magnetic current
    
    # Convert Vector objects to PyTorch tensors with shape (3, N_src)
    r_src = T.stack([
        T.tensor(face1.x.ravel(), dtype=T.float64, device=device),
        T.tensor(face1.y.ravel(), dtype=T.float64, device=device),
        T.tensor(face1.z.ravel(), dtype=T.float64, device=device)
    ], dim=0)
    
    r_obs = T.stack([
        T.tensor(face2.x.ravel(), dtype=T.float64, device=device),
        T.tensor(face2.y.ravel(), dtype=T.float64, device=device),
        T.tensor(face2.z.ravel(), dtype=T.float64, device=device)
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

def scalarproduct(scalar, vec_obj):
    """
    Backward compatibility wrapper for scalar multiplication.
    
    Maps old scalarproduct(k, v) to v * k using Vector objects.
    
    Args:
        scalar: Numeric scalar value
        vec_obj: Vector object
    
    Returns:
        New Vector object with scaled components
    """
    return vec_obj * scalar


def crossproduct(v1, v2):
    """
    Backward compatibility wrapper for cross product.
    
    Maps old crossproduct(v1, v2) to cross(v1, v2) using Vector objects.
    
    Args:
        v1, v2: Vector objects
    
    Returns:
        New Vector object with cross product result
    """
    return cross(v1, v2)


'''2.2 calculate the far-field beam'''    
def PO_far(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,parallel=True,device =T.device('cpu')):
   # output field:
    Field_E=vector()
    Field_H=vector()  
    Je_in=scalarproduct(2,crossproduct(face1_n,Field_in_H))
    JE=np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,-1)
    print(JE.shape)
    ''' calculate the field including the large matrix'''
    #@njit(parallel=parallel)
    def calculus1(x1,y1,z1,x2,y2,z2,N,ds): 
        Field_E_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_E_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_E_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        rp=np.zeros((3,x1.size))
        rp[0,:] = x1
        rp[1,:] = y1
        rp[2,:] = z1
        for i in prange(x2.size):
            r = np.array([[x2[i]],[y2[i]],[z2[i]]])
            phase = k*np.sum(rp*r,axis=0)
            Ee = (JE-np.sum(JE*r,axis = 0) * r) * np.exp(1j*phase)*k**2
            Ee = np.sum(Ee*N*ds,axis=-1)*(-1j*Z0/4/np.pi)
            #print(Ee)
            Field_E_x[i] = Ee[0]
            Field_E_y[i] = Ee[1]
            Field_E_z[i] = Ee[2]
        #print(Field_E_x)
        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z
    Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus1(face1.x,face1.y,face1.z,
                                                                          face2.x.ravel(),face2.y.ravel(),face2.z.ravel(),
                                                                          face1_n.N.ravel(),face1_dS.ravel())
    #print(Field_E.x)
    return Field_E,Field_H
        


    

def PO_far_GPU2(face1,face1_n,face1_dS,
               face2,
               Field_in_E,
               Field_in_H,
               k,
               device =T.device('cuda')):
    # output field:
    N_f = face2.x.size
    Field_E = vector()
    Field_E.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.z = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H = vector()
    Field_H.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.z = T.zeros(N_f, dtype=T.complex128, device=device)

    ds = face1_dS * face1_n.N * k**2 /4/np.pi


    J_in=scalarproduct(2*ds,crossproduct(face1_n,Field_in_H))
    JE=T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1),
                dtype = T.complex128).to(device)
    
    J_in = scalarproduct(2*ds, crossproduct(face1_n, Field_in_E))
    JM = T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device).contiguous()
    
    # convert surface points into tensor
    Surf1 = T.stack([T.tensor(face1.x.ravel()), 
                     T.tensor(face1.y.ravel()), 
                     T.tensor(face1.z.ravel())], dim=0).reshape(3,1,-1)
    Surf1 = Surf1.contiguous().to(device)
    Surf2 = T.stack([T.tensor(face2.x.ravel()), 
                     T.tensor(face2.y.ravel()), 
                     T.tensor(face2.z.ravel())], dim=0).reshape(3,-1,1)
    Surf2 = Surf2.contiguous().to(device)

    def calculate_fields(s2,K):
        """
        Helper function to calculate fields for a batch of points.
        """
        # Compute R and r
        Phase = k * T.sum((s2 * Surf1).contiguous(),axis = 0).contiguous()
        Phase =  T.exp(1j * Phase) #* ds
        # Electric field calculation
        ee = T.sum( JM * Phase, axis = -1 ).contiguous()
        r = s2.reshape(3,-1).contiguous().to(dtype = T.complex128)
        Ee = 1j * T.cross(r, ee, dim=0).contiguous()
        # Magnetic field calculation
        he = T.sum( JE * Phase, axis = -1 ).contiguous()
        He = -1j /Z0 * T.cross(r, he, dim=0).contiguous()

        return Ee, He
    
    if device == T.device('cuda'):
        # Get total, allocated, and reserved memory
        total_memory = T.cuda.get_device_properties(0).total_memory
        allocated_memory = T.cuda.memory_allocated(0)
        reserved_memory = T.cuda.memory_reserved(0)

        # Calculate free memory
        free_memory = total_memory - reserved_memory

        # Adjust batch size based on free memory
        element_size = JE.element_size() * JE.nelement()
        batch_size = int(free_memory / element_size / 6)
    else:
        batch_size = os.cpu_count() * 30

    print(f"Batch size: {batch_size}")
    N = face2.x.size
    num_batches = N // batch_size

    # Process batches
    #print(Surf2.device, Surf1.device, k_n.device)
    #print(Surf2.is_contiguous(), Surf1.is_contiguous(), k_n.is_contiguous())

    with T.no_grad():
        for i in tqdm(range(num_batches),mininterval=5):
            start = i * batch_size
            end = (i + 1) * batch_size
            #idx = T.arange(start, end, device ='cuda')
            Ee, He = calculate_fields(Surf2[:,start:end,:].contiguous() ,k)

            Field_E.x[start:end] = Ee[0, :]
            Field_E.y[start:end] = Ee[1, :]
            Field_E.z[start:end] = Ee[2, :]
            Field_H.x[start:end] = He[0, :]
            Field_H.y[start:end] = He[1, :]
            Field_H.z[start:end] = He[2, :]
                
        # Process remaining points
        if N % batch_size != 0:
            start = num_batches * batch_size
            Ee, He = calculate_fields(Surf2[:,start:,:].contiguous(),k)
            Field_E.x[start:] = Ee[0, :]
            Field_E.y[start:] = Ee[1, :]
            Field_E.z[start:] = Ee[2, :]
            Field_H.x[start:] = He[0, :]
            Field_H.y[start:] = He[1, :]
            Field_H.z[start:] = He[2, :]
    # Move tensors back to CPU only once    
    Field_E.Tensor2np()
    Field_H.Tensor2np()

    T.cuda.empty_cache()
    T.cuda.synchronize()
    return Field_E, Field_H
 

# %%
