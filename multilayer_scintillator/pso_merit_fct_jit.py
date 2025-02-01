import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np
from numba import jit
from itertools import product
from mpi4py import MPI

comm = MPI.COMM_WORLD

def transmitted_intensity_reconstruction(RGB_noise,
                                         sensRGB_interp,
                                         wvl_hist_temp,
                                         gamma_block,
                                         energy_bins,
                                         Ntrain,
                                         raw_clip,
                                         ):

    Di = (sensRGB_interp.T @ wvl_hist_temp)/raw_clip
    
    ##########
#    D = np.zeros((Ntrain*3, Ntrain*energy_bins))
#    for i in range(Ntrain):
#        D[3*i:3*(i+1),energy_bins*i:energy_bins*(i+1)] = Di
#    
#    gamma = np.zeros((Ntrain*energy_bins, Ntrain*energy_bins))
#    for i in range(Ntrain):
#        gamma[energy_bins*i:energy_bins*(i+1),energy_bins*i:energy_bins*(i+1)] = gamma_block
#        
#    I_reconstr_direct = np.linalg.inv(D.T @ D + gamma) @ D.T @ RGB_noise
    ##########
    
    RGB_noise = RGB_noise.reshape(Ntrain, 3).T
    
    quo, rem = divmod(Ntrain, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
    RGB_per_proc = RGB_noise[:,data_disp[comm.rank]:data_disp[comm.rank+1]]

    # Direct Inversion
    I_per_proc = np.zeros((data_size[comm.rank], energy_bins))
    for n in range(data_size[comm.rank]):
        I_per_proc[n,:] = np.linalg.inv(Di.T @ Di + gamma_block) @ Di.T @ RGB_per_proc[:,n]
    
    data_size_temp = data_size*energy_bins
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    I_temp = np.zeros(Ntrain*energy_bins)
    comm.Allgatherv(I_per_proc.reshape(-1), [I_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
    I_reconstr = I_temp.copy()
    
#    if comm.rank == 0:
#        np.savez(directory + "/data/parallel_I_test", I_reconstr=I_reconstr, I_reconstr_direct=I_reconstr_direct)
#    assert False
    
    return I_reconstr, Di, gamma_block

def jac_step1_reg_obj(gamma_vec,
                      RGB_noise,
                      sensRGB_interp,
                      wvl_hist_temp,
                      xraySrc_fwd,
                      energy_bins,
                      Ntrain,
                      raw_clip,
                      ):
                      
    I_reconstr, D, gamma = transmitted_intensity_reconstruction(RGB_noise,
                                                                sensRGB_interp,
                                                                wvl_hist_temp,
                                                                gamma_vec.reshape(energy_bins, energy_bins),
                                                                energy_bins,
                                                                Ntrain,
                                                                raw_clip,
                                                                )

    A1 = D.T @ D + gamma
    A1_inv = np.linalg.inv(A1)
    grad = np.zeros((Ntrain*energy_bins, energy_bins, energy_bins))
    
    i_list = np.arange(energy_bins)
    j_list = np.arange(energy_bins)
    ij = np.array(list(product(i_list, j_list)))
    
    quo, rem = divmod(energy_bins**2, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
    ij_per_proc = ij[data_disp[comm.rank]:data_disp[comm.rank+1],:]
    
    grad_per_proc = np.zeros((Ntrain*energy_bins, data_size[comm.rank]))
    for n in range(data_size[comm.rank]):
        i = ij_per_proc[n,0]
        j = ij_per_proc[n,1]
        
        P = np.zeros((energy_bins, energy_bins))
        P[i,j] = 1
        P_block_diag = np.zeros((Ntrain*energy_bins, Ntrain*energy_bins))
        for k in range(Ntrain):
            P_block_diag[energy_bins*k:energy_bins*(k+1),energy_bins*k:energy_bins*(k+1)] = P

        result_vector = -A1_inv @ P_block_diag @ A1_inv @ D.T @ RGB_noise
    
        grad_per_proc[:,n] = result_vector
    
    data_size_temp = data_size*Ntrain*energy_bins
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    grad_temp = np.zeros(Ntrain*energy_bins**3)
    comm.Allgatherv(grad_per_proc.T.reshape(-1), [grad_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
    grad = grad_temp.reshape(energy_bins**2, Ntrain*energy_bins).T
    
    xraySrc_temp = xraySrc_fwd.reshape(Ntrain, energy_bins)
    weights = np.tile(1/np.sum(xraySrc_temp**2, axis=1), (energy_bins,1)).T.reshape(-1)
    d_loss = 2*(I_reconstr - xraySrc_fwd)
    grad_loss = d_loss.reshape(1,-1) @ grad
    
    return grad_loss.reshape(-1)

def jac_step1_reg_obj_diag(gamma_diag,
                           RGB_noise,
                           sensRGB_interp,
                           wvl_hist_temp,
                           xraySrc_fwd,
                           energy_bins,
                           Ntrain,
                           raw_clip,
                           ):
                      
    I_reconstr, Di, gamma_block = transmitted_intensity_reconstruction(RGB_noise,
                                                                       sensRGB_interp,
                                                                       wvl_hist_temp,
                                                                       np.diag(gamma_diag),
                                                                       energy_bins,
                                                                       Ntrain,
                                                                       raw_clip,
                                                                       )

    RGB_noise = RGB_noise.reshape(Ntrain, 3).T

    A1 = Di.T @ Di + gamma_block
    A1_inv = np.linalg.inv(A1)
    
    i_list = np.arange(energy_bins)
    j_list = np.arange(Ntrain)
    ij = np.array(list(product(i_list, j_list)))
    
    quo, rem = divmod(Ntrain*energy_bins, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
    ij_per_proc = ij[data_disp[comm.rank]:data_disp[comm.rank+1],:]
    
    grad_per_proc = np.zeros((data_size[comm.rank], energy_bins))
    for n in range(data_size[comm.rank]):
        i = ij_per_proc[n,0]
        j = ij_per_proc[n,1]
        
        P = np.zeros((energy_bins, energy_bins))
        P[i,i] = 1
        
        result_vector = -A1_inv @ P @ A1_inv @ Di.T @ RGB_noise[:,j]
        
        grad_per_proc[n,:] = result_vector
    
    data_size_temp = data_size*energy_bins
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    grad_temp = np.zeros(Ntrain*energy_bins**2)
    comm.Allgatherv(grad_per_proc.reshape(-1), [grad_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
    grad = grad_temp.reshape(energy_bins, Ntrain, energy_bins).transpose(1,2,0).reshape(Ntrain*energy_bins, energy_bins)
    
    xraySrc_temp = xraySrc_fwd.reshape(Ntrain, energy_bins)
    weights = np.tile(1/np.sum(xraySrc_temp**2, axis=1), (energy_bins,1)).T.reshape(-1)
    d_loss = 2*weights*(I_reconstr - xraySrc_fwd)
    grad_loss = d_loss.reshape(1,-1) @ grad
    
    return grad_loss.reshape(-1)
    
def element_thickness_reconstruction(T_reconstr,
                                     linAttCoeff,
                                     gamma_block,
                                     energy_bins,
                                     Nsample,
                                     Ntrain,
                                     ):

    ##########
#    mu = np.zeros((Ntrain*energy_bins, Ntrain*Nsample))
#    for i in range(Ntrain):
#        mu[energy_bins*i:energy_bins*(i+1),Nsample*i:Nsample*(i+1)] = -linAttCoeff
#    
#    gamma = np.zeros((Ntrain*Nsample, Ntrain*Nsample))
#    for i in range(Ntrain):
#        gamma[Nsample*i:Nsample*(i+1),Nsample*i:Nsample*(i+1)] = gamma_block
#    
#    d_reconstr_direct = np.linalg.inv(mu.T @ mu + gamma) @ mu.T @ np.log(T_reconstr)
    ##########
    
    T_reconstr = T_reconstr.reshape(Ntrain, energy_bins).T
    
    quo, rem = divmod(Ntrain, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
    T_per_proc = T_reconstr[:,data_disp[comm.rank]:data_disp[comm.rank+1]]
    
    # Direct Inversion
    d_per_proc = np.zeros((data_size[comm.rank], Nsample))
    for n in range(data_size[comm.rank]):
        d_per_proc[n,:] = np.linalg.inv(linAttCoeff.T @ linAttCoeff + gamma_block) @ -linAttCoeff.T @ np.log(T_per_proc[:,n])
    
    data_size_temp = data_size*Nsample
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    d_temp = np.zeros(Ntrain*Nsample)
    comm.Allgatherv(d_per_proc.reshape(-1), [d_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
    d_reconstr = d_temp.copy()
    
#    if comm.rank == 0:
#        np.savez(directory + "/data/debug_thickness", d_reconstr=d_reconstr, d_reconstr_direct=d_reconstr_direct)
    
    return d_reconstr, linAttCoeff, gamma_block

def jac_step2_reg_obj(gamma_vec,
                      T_reconstr,
                      linAttCoeff,
                      training_data,
                      energy_bins,
                      Nsample,
                      Ntrain,
                      ):
                      
    d_reconstr, mu, gamma = element_thickness_reconstruction(T_reconstr,
                                                             linAttCoeff,
                                                             gamma_vec.reshape(Nsample, Nsample),
                                                             energy_bins,
                                                             Nsample,
                                                             Ntrain,
                                                             )

    A1 = mu.T @ mu + gamma
    A1_inv = np.linalg.inv(A1)
    grad = np.zeros((Ntrain*Nsample, Nsample, Nsample))
    
    i_list = np.arange(Nsample)
    j_list = np.arange(Nsample)
    ij = np.array(list(product(i_list, j_list)))
    
    quo, rem = divmod(Nsample**2, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
    ij_per_proc = ij[data_disp[comm.rank]:data_disp[comm.rank+1],:]
    
    grad_per_proc = np.zeros((Ntrain*Nsample, data_size[comm.rank]))
    for n in range(data_size[comm.rank]):
        i = ij_per_proc[n,0]
        j = ij_per_proc[n,1]
        
        P = np.zeros((Nsample, Nsample))
        P[i,j] = 1
        P_block_diag = np.zeros((Ntrain*Nsample, Ntrain*Nsample))
        for k in range(Ntrain):
            P_block_diag[Nsample*k:Nsample*(k+1),Nsample*k:Nsample*(k+1)] = P

        result_vector = -A1_inv @ P_block_diag @ A1_inv @ mu.T @ np.log(T_reconstr)
    
        grad_per_proc[:,n] = result_vector
    
    data_size_temp = data_size*Ntrain*Nsample
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    grad_temp = np.zeros(Ntrain*Nsample**3)
    comm.Allgatherv(grad_per_proc.T.reshape(-1), [grad_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
    grad = grad_temp.reshape(Nsample**2, Ntrain*Nsample).T
    # grad = grad.reshape(Nsample**2, Ntrain, Nsample)[:,:,-1].T # only compute for iodine
    
    # d_loss = 2*(d_reconstr.reshape(Ntrain, Nsample)[:,-1] - training_data) # only compute for iodine
    weights = 1/training_data
    weights[training_data==0] = np.max(weights[weights!=np.inf])
    weights[(np.arange(weights.size)%Nsample)!=Nsample-1] = 0
    d_loss = 2*(d_reconstr - training_data)
    grad_loss = d_loss.reshape(1,-1) @ grad
    
    return grad_loss.reshape(-1)

def jac_step2_reg_obj_diag(gamma_diag,
                           T_reconstr,
                           linAttCoeff,
                           training_data,
                           energy_bins,
                           Nsample,
                           Ntrain,
                           ):
                      
    d_reconstr, linAttCoeff, gamma_block = element_thickness_reconstruction(T_reconstr,
                                                                            linAttCoeff,
                                                                            np.diag(gamma_diag),
                                                                            energy_bins,
                                                                            Nsample,
                                                                            Ntrain,
                                                                            )

    T_reconstr = T_reconstr.reshape(Ntrain, energy_bins).T

    A1 = linAttCoeff.T @ linAttCoeff + gamma_block
    A1_inv = np.linalg.inv(A1)
    grad = np.zeros((Ntrain, Nsample, Nsample))
    
    i_list = np.arange(Nsample)
    j_list = np.arange(Ntrain)
    ij = np.array(list(product(i_list, j_list)))
    
    quo, rem = divmod(Nsample*Ntrain, comm.size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
    ij_per_proc = ij[data_disp[comm.rank]:data_disp[comm.rank+1],:]
    
    grad_per_proc = np.zeros((data_size[comm.rank], Nsample))
    for n in range(data_size[comm.rank]):
        i = ij_per_proc[n,0]
        j = ij_per_proc[n,1]
        
        P = np.zeros((Nsample, Nsample))
        P[i,i] = 1

        result_vector = -A1_inv @ P @ A1_inv @ -linAttCoeff.T @ np.log(T_reconstr[:,j])
    
        grad_per_proc[n,:] = result_vector
    
    data_size_temp = data_size*Nsample
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    grad_temp = np.zeros(Ntrain*Nsample**2)
    comm.Allgatherv(grad_per_proc.reshape(-1), [grad_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
    grad = grad_temp.reshape(Nsample, Ntrain, Nsample).transpose(1,2,0).reshape(Ntrain*Nsample, Nsample)

    d_loss = 2*(d_reconstr - training_data)
    grad_loss = d_loss.reshape(1,-1) @ grad
    
    return grad_loss.reshape(-1)

@jit(nopython=True, cache=True)
def iodine_thickness_reconstruction(T_reconstr,
                                    linAttCoeff,
                                    gamma_block,
                                    energy_bins,
                                    Nsample,
                                    Ntrain,
                                    ):

    mu = np.zeros((Ntrain*energy_bins, Ntrain*Nsample))
    for i in range(Ntrain):
        mu[energy_bins*i:energy_bins*(i+1),Nsample*i:Nsample*(i+1)] = -linAttCoeff
    
    gamma = np.zeros((Ntrain*Nsample, Ntrain*Nsample))
    for i in range(Ntrain):
        gamma[Nsample*i:Nsample*(i+1),Nsample*i:Nsample*(i+1)] = gamma_block
    
    # Direct Inversion
    d_reconstr = np.linalg.inv(mu.T @ mu + gamma) @ mu.T @ np.log(T_reconstr)
    
    return d_reconstr, mu, gamma