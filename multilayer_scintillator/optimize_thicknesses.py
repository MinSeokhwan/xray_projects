import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
#import autograd.numpy as npa
#from autograd import grad
from scipy.optimize import minimize, Bounds
from numba import jit
import util.read_txt as txt

nlayers = 1
nbin = 3

energy_list = np.linspace(16, 67, 52)
mu = np.zeros((energy_list.size, nlayers))
#raw = txt.read_txt(directory + "/data/linAttCoeffZnSeTe")
#mu[:,0] = np.interp(energy_list, raw[:,0], raw[:,1])
#raw = txt.read_txt(directory + "/data/linAttCoeffYAGCe")
#mu[:,0] = np.interp(energy_list, raw[:,0], raw[:,1])
#raw = txt.read_txt(directory + "/data/linAttCoeffCsITl")
#mu[:,1] = np.interp(energy_list, raw[:,0], raw[:,1])
#raw = txt.read_txt(directory + "/data/linAttCoeffLYSOCe")
#mu[:,2] = np.interp(energy_list, raw[:,0], raw[:,1])
#raw = txt.read_txt(directory + "/data/linAttCoeffGSOCe")
#mu[:,1] = np.interp(energy_list, raw[:,0], raw[:,1])
#raw = txt.read_txt(directory + "/data/linAttCoeffGadoxTb")
#mu[:,1] = np.interp(energy_list, raw[:,0], raw[:,1])
raw = txt.read_txt(directory + "/data/linAttCoeffNaITl")
mu[:,0] = np.interp(energy_list, raw[:,0], raw[:,1])
#raw = txt.read_txt(directory + "/data/linAttCoeffBGO")
#mu[:,1] = np.interp(energy_list, raw[:,0], raw[:,1])

def cost_fct(thickness, mu, matrix=False):
    M_out = np.zeros((nlayers, energy_list.size))
    M = np.zeros((nlayers, nbin)) # layer x bin
    T0 = 1
    for i in range(nlayers):
        T = T0*np.exp(-mu[:,i]*thickness[i])
        A = T0 - T
        M_out[i,:] = A
        for j in range(nbin):
            M[i,j] = np.sum(A[energy_bin_ind[j]:energy_bin_ind[j+1]])
        T0 = T
    
#    cost = np.zeros(3)
#    cost[0] = np.min(M_out[0,energy_bin_ind[0]:energy_bin_ind[1]]/(M_out[1,energy_bin_ind[0]:energy_bin_ind[1]] + M_out[2,energy_bin_ind[0]:energy_bin_ind[1]]))
#    cost[1] = np.min(M_out[1,energy_bin_ind[1]:energy_bin_ind[2]]/(M_out[0,energy_bin_ind[1]:energy_bin_ind[2]] + M_out[2,energy_bin_ind[1]:energy_bin_ind[2]]))
#    cost[2] = np.min(M_out[2,energy_bin_ind[2]:energy_bin_ind[3]]/(M_out[0,energy_bin_ind[2]:energy_bin_ind[3]] + M_out[1,energy_bin_ind[2]:energy_bin_ind[3]]))
#    
#    cost = np.min(cost)
    
    # cost = np.min((M[0,0]/(M[0,1] + M[0,2]), M[1,1]/(M[1,0] + M[1,2]), M[2,2]/(M[2,1] + M[2,0]))))
    
    cost = np.zeros(nbin)
    for i in range(nbin):
        cost[i] = M[i,i]/(np.sum(M[:,i]) - M[i,i] + 1e-8)
    cost = np.min(cost)
    
    if matrix:
        return M, M_out
    else:
        return -cost

@jit(nopython=True, cache=True)
def accuracy_cost_fct_jit(thickness, mu, output=False):
    z = np.linspace(0, np.sum(thickness), 3001) # less than 1um resolution
    
    # Absorption per unit length
    A = np.zeros((mu.shape[0], z.size))
    if nlayers == 3:
        for nz in range(z.size):
            if z[nz] < thickness[0]:
                A[:,nz] = mu[:,0]*np.exp(-mu[:,0]*z[nz])
            elif z[nz] < np.sum(thickness[:2]):
                A[:,nz] = mu[:,1]*np.exp(-mu[:,0]*thickness[0] - mu[:,1]*(z[nz] - thickness[0]))
            else:
                A[:,nz] = mu[:,2]*np.exp(-mu[:,0]*thickness[0] - mu[:,1]*thickness[1] - mu[:,2]*(z[nz] - np.sum(thickness[:2])))
    elif nlayers == 2:
        for nz in range(z.size):
            if z[nz] < thickness[0]:
                A[:,nz] = mu[:,0]*np.exp(-mu[:,0]*z[nz])
            else:
                A[:,nz] = mu[:,1]*np.exp(-mu[:,0]*thickness[0] - mu[:,1]*(z[nz] - thickness[0]))
    elif nlayers == 1:
        for nz in range(z.size):
            A[:,nz] = mu[:,0]*np.exp(-mu[:,0]*z[nz])
    
    # Probability of absorption at a given depth given the incident energy
    PzE = A/np.sum(A, axis=1)[:,np.newaxis]
    
    # Probability of classification into a certain energy given the absorption depth
    PEz = A/np.sum(A, axis=0)[np.newaxis,:]
    
    # Probability of classification into a certain energy given the incident energy
    P = PzE @ PEz.T
    
    # Confusion Matrix
    M = np.zeros((nbin, nbin))
    for i in range(nbin):
        for j in range(nbin):
            M[i,j] = np.sum(P[energy_bin_ind[i]:energy_bin_ind[i+1],energy_bin_ind[j]:energy_bin_ind[j+1]])
    M /= np.sum(M, axis=1)[:,np.newaxis]
    
    # Classification Accuracy
    acc = np.trace(M)/np.sum(M)
    # acc = np.min(np.diag(M))
    # acc = np.trace(P)/np.sum(P)
    
    return -acc

def accuracy_cost_fct(thickness, mu):
    z = np.linspace(0, np.sum(thickness), 3001) # less than 1um resolution
    
    # Absorption per unit length
    A = np.zeros((mu.shape[0], z.size))
    if nlayers == 3:
        for nz in range(z.size):
            if z[nz] < thickness[0]:
                A[:,nz] = mu[:,0]*np.exp(-mu[:,0]*z[nz])
            elif z[nz] < np.sum(thickness[:2]):
                A[:,nz] = mu[:,1]*np.exp(-mu[:,0]*thickness[0] - mu[:,1]*(z[nz] - thickness[0]))
            else:
                A[:,nz] = mu[:,2]*np.exp(-mu[:,0]*thickness[0] - mu[:,1]*thickness[1] - mu[:,2]*(z[nz] - np.sum(thickness[:2])))
    elif nlayers == 2:
        for nz in range(z.size):
            if z[nz] < thickness[0]:
                A[:,nz] = mu[:,0]*np.exp(-mu[:,0]*z[nz])
            else:
                A[:,nz] = mu[:,1]*np.exp(-mu[:,0]*thickness[0] - mu[:,1]*(z[nz] - thickness[0]))
    elif nlayers == 1:
        for nz in range(z.size):
            A[:,nz] = mu[:,0]*np.exp(-mu[:,0]*z[nz])
    
    # Probability of absorption at a given depth given the incident energy
    PzE = A/np.sum(A, axis=1)[:,np.newaxis]
    
    # Probability of classification into a certain energy given the absorption depth
    PEz = A/np.sum(A, axis=0)[np.newaxis,:]
    
    # Probability of classification into a certain energy given the incident energy
    P = PzE @ PEz.T
    
    # Confusion Matrix
    M = np.zeros((nbin, nbin))
    for i in range(nbin):
        for j in range(nbin):
            M[i,j] = np.sum(P[energy_bin_ind[i]:energy_bin_ind[i+1],energy_bin_ind[j]:energy_bin_ind[j+1]])
    M /= np.sum(M, axis=1)[:,np.newaxis]
    
    # Classification Accuracy
    acc = np.trace(M)/np.sum(M)
    # acc = np.trace(P)/np.sum(P)

    return z, mu, A, PzE, PEz, P, M, acc

energy_bin_ind = np.array([0,17,34,52]).astype(int)
cost_all = np.zeros(1000)
thickness_all = np.zeros((nlayers, 1000))
for i in range(1000):
    # thickness0 = np.exp((np.log(0.2) - np.log(0.001))*np.random.rand(nlayers)) + 0.001
    thickness0 = np.exp((np.log(0.001) - np.log(0.03))*np.random.rand(nlayers) + np.log(0.03))
    bnd = Bounds(lb=0.001, ub=0.03)
    result = minimize(accuracy_cost_fct_jit, thickness0, args=(mu), jac='3-point', method='L-BFGS-B', bounds=bnd)
    
    cost_all[i] = result.fun
    thickness_all[:,i] = result.x
ind_best = np.argmin(cost_all)
z, mu, A, PzE, PEz, P, M, acc = accuracy_cost_fct(thickness_all[:,ind_best], mu)
#z2, mu2, A2, PzE2, PEz2, P2, M2, acc2 = accuracy_cost_fct(np.array([0.00538629,0.010233,0.0304055]), mu)
print('Optimization Complete', flush=True)
np.savez(directory + "/data/optimize_thickness_NaI_67keV_d0_3mm_accuracy_cost", z=z, mu=mu, A=A, PzE=PzE, PEz=PEz, P=P, M=M, acc=acc, thickness=thickness_all[:,ind_best])#,
#         z2=z2, mu2=mu2, A2=A2, PzE2=PzE2, PEz2=PEz2, P2=P2, M2=M2, acc2=acc2)

#nE1 = 1
#nE2 = 1
#Estep = 17
#cost_sweep = np.zeros((nE1, nE2))
#thickness_sweep = np.zeros((nlayers, nE1, nE2))
#M_sweep = np.zeros((nlayers, nbin, nE1, nE2))
#M_out_sweep = np.zeros((nlayers, energy_list.size, nE1, nE2))
#for E1 in range(nE1):
#    energy1 = energy_list[Estep*E1+Estep]
#    print('Energy 1: ' + str(energy1) + ' keV', flush=True)
#    for E2 in range(nE2):
#        energy2 = energy_list[Estep*E2+2*Estep]
#        if energy2 > energy1:
#            print('\tEnergy 2: ' + str(energy2) + ' keV', flush=True)
#            energy_bin_ind = np.array([0,energy1-Estep,energy2-Estep,90]).astype(int)
#            
#            cost_all = np.zeros(1000)
#            thickness_all = np.zeros((nlayers, 1000))
#            M_all = np.zeros((nlayers, nbin, 1000))
#            M_out_all = np.zeros((nlayers, energy_list.size, 1000))
#            for i in range(1000):
#                thickness0 = (0.1 - 0.001)*np.random.rand(nlayers) + 0.001
#                bnd = Bounds(lb=0.001, ub=0.1)
#                result = minimize(cost_fct, thickness0, args=(mu), method='L-BFGS-B', bounds=bnd)
#                
#                cost_all[i] = result.fun
#                thickness_all[:,i] = result.x
#                M_all[:,:,i], M_out_all[:,:,i] = cost_fct(result.x, mu, matrix=True)
#            
#            best_ind = np.argmin(cost_all)
#            cost_sweep[E1,E2] = cost_all[best_ind]
#            thickness_sweep[:,E1,E2] = thickness_all[:,best_ind]
#            M_sweep[:,:,E1,E2] = M_all[:,:,best_ind]
#            M_out_sweep[:,:,E1,E2] = M_out_all[:,:,best_ind]
#
##M_all[:,:,-2], M_out_all[:,:,-2] = cost_fct(np.array([0.0050783,0.0012911,0.0037092]), mu, matrix=True)
##M_all[:,:,-1], M_out_all[:,:,-1] = cost_fct(np.array([0.00538629,0.010233,0.0304055]), mu, matrix=True)
#
#print('Sweep Complete', flush=True)
#np.savez(directory + "/data/optimize_thickness_ZnSe_CsI_GSO_67keV_coarse_sweep", cost=cost_sweep, thickness=thickness_sweep, M=M_sweep, M_out=M_out_sweep)