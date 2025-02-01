import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
import util.read_txt as txt
from scipy.io import loadmat
from scipy.optimize import minimize
import multilayer_scintillator.xray_spectrum_generator as xray

from mpi4py import MPI

comm = MPI.COMM_WORLD

def generate(filename, energy_list, Ntrain):
    if filename == 'AM':
        voxelDim = np.array([2.137,2.137,8])/10 # in cm
    elif filename == 'AF':
        voxelDim = np.array([1.775,1.775,4.84])/10 # in cm

    density = txt.read_txt(directory + "/data/ICRP110/elemental_density")
    phantom = loadmat(directory + "/data/ICRP110/" + filename + "_3D")['phantom']
    media = txt.read_txt(directory + "/data/ICRP110/" + filename + "_media")/100
    organs = txt.read_txt(directory + "/data/ICRP110/" + filename + "_organs")
    blood = txt.read_txt(directory + "/data/ICRP110/" + filename + "_blood")
    
    organs[[8,9,10,11,87,95,97],0] = 55 # 1% Gd + blood
    
    energy_bins = energy_list.size
    Nx, Ny, Nz = phantom.shape
    N_organs = organs.shape[0]
    N_media = media.shape[0]
    N_elements = media.shape[1]
    
    element_list = ['H','C','N','O','Na','Mg','P','S','Cl','K','Ca','Fe','I','Gd']
    linAttCoeff_element = np.zeros((energy_bins, N_elements))
    for ne in range(N_elements):
        raw = txt.read_txt(directory + '/data/linAttCoeff' + element_list[ne])
        linAttCoeff_element[:,ne] = np.interp(energy_list, raw[:,0], raw[:,1], left=0, right=0)
    
    scale = np.sqrt(np.sum(linAttCoeff_element**2, axis=0))
    linAttCoeff_element_scaled = linAttCoeff_element/scale[np.newaxis,:]
    
    U, S, Vh = np.linalg.svd(linAttCoeff_element_scaled[:,:-1])
    linAttCoeff_element_svd = np.hstack((U[:,:2], linAttCoeff_element_scaled[:,-1].reshape(-1,1)))
    linAttCoeff_element_svd[:,:2] *= np.sqrt(np.sum(linAttCoeff_element_svd[:,-1]**2))
    
#    combination_factors = np.zeros((N_elements, 2))
#    for ne in range(N_elements-1):
#        result = minimize(obj_lsq, x0=np.zeros(2), args=(linAttCoeff_element_svd[:,:2], linAttCoeff_element[:,ne]),
#                          method='BFGS', jac=jac_lsq)
#        combination_factors[ne,:] = result.x
#    
#    np.savez(directory + '/data/svd_data', combination_factors=combination_factors, linAttCoeff_element_svd=linAttCoeff_element_svd, linAttCoeff_element=linAttCoeff_element)
#    assert False
    
#    linAttCoeff_media = (media @ linAttCoeff_element.T).T
#    density_media = media @ density
#    
#    linAttCoeff_organs = np.zeros((energy_bins, N_organs))
#    for no in range(N_organs):
#        linAttCoeff_organs[:,no] = linAttCoeff_media[:,int(organs[no,0]-1)]*organs[no,1]/density_media[int(organs[no,0]-1)]
    
    organ_thickness_per_pixel = np.zeros((N_organs, Nx, Nz))
    for no in range(N_organs):
        organ_thickness_per_pixel[no,:,:] = voxelDim[1]*np.sum(phantom==no+1, axis=1)
    
    element_thickness_per_pixel = np.zeros((N_elements, Nx, Nz))
    for no in range(N_organs):
        percent_elements = media[int(organs[no,0]-1),:]
        element_thickness_per_pixel += percent_elements[:,np.newaxis,np.newaxis]*organ_thickness_per_pixel[no,:,:][np.newaxis,:,:]
    
#    iodine_data = element_thickness_per_pixel.reshape(N_elements,-1)[:,element_thickness_per_pixel.reshape(N_elements,-1)[-1,:]>0].T
#    non_iodine_data = element_thickness_per_pixel.reshape(N_elements,-1)[:,element_thickness_per_pixel.reshape(N_elements,-1)[-1,:]==0].T
#    iodine_percentage = iodine_data[:,-1]/np.sum(iodine_data, axis=1)
#    if comm.rank == 0:
#        print('\n### Training Data Description', flush=True)
#        print('\t | Minimum non-zero iodine percentage: ' + str(100*np.min(iodine_percentage)) + '%', flush=True)
#        print('\t | Maximum non-zero iodine percentage: ' + str(100*np.max(iodine_percentage)) + '%', flush=True)

    kidneys = np.argwhere(np.sum((phantom>=89)*(phantom<=94), axis=1) > 0)
    
    limits = np.zeros(4)
    limits[:2] = np.min(kidneys, axis=0)
    limits[2:] = np.max(kidneys, axis=0)
    limits = limits.astype(int)
    
    sampled_data = element_thickness_per_pixel[:,limits[0]:limits[2],limits[1]:limits[3]].reshape(N_elements,-1).T
    
    np.random.seed(0)
#    training_data = np.vstack((iodine_data[np.random.randint(0, iodine_data.shape[0], size=iodine_data.shape[0])[:int(Ntrain/2)],:],
#                               non_iodine_data[np.random.randint(0, non_iodine_data.shape[0], size=non_iodine_data.shape[0])[:int(Ntrain/2)],:]))
    # training_data = iodine_data[np.random.randint(0, iodine_data.shape[0], size=iodine_data.shape[0])[:Ntrain],:]*scale[np.newaxis,:]
    training_data = sampled_data[np.random.randint(0, sampled_data.shape[0], size=sampled_data.shape[0])[:Ntrain],:]*scale[np.newaxis,:]
    np.random.seed()
    
#    linAttCoeff_train = linAttCoeff_element[:,:-1] @ training_data[:,:-1].T # energy_bins x Ntrain
#    combination_factors = np.zeros((N_train, 3))
#    for nt in range(N_train):
#        result = minimize(obj_lsq, x0=np.zeros(2), args=(linAttCoeff_element_svd[:,:2], linAttCoeff_train[:,nt]),
#                          method='BFGS', jac=jac_lsq)
#        combination_factors[nt,:2] = result.x
#    combination_factors[:,-1] = training_data[:,-1]
    
    return linAttCoeff_element, linAttCoeff_element_scaled, training_data, sampled_data.T.reshape(N_elements,limits[2]-limits[0],-1), element_thickness_per_pixel, linAttCoeff_element_svd, scale

def obj_lsq(x, linAttCoeff_element_svd, linAttCoeff_train):
    cost = np.sum((linAttCoeff_element_svd @ x - linAttCoeff_train)**2)
    
    return cost

def jac_lsq(x, linAttCoeff_element_svd, linAttCoeff_train):
    jac = 2*linAttCoeff_element_svd.T @ (linAttCoeff_element_svd @ x - linAttCoeff_train)
    
    return jac