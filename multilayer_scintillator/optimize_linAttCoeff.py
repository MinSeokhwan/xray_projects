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

nlayers = 3
nbin = 3
energy_list = np.linspace(16, 67, 52)

@jit(nopython=True, cache=True)
def accuracy_cost_fct_jit(coeff, energy_list, output=False):
    thickness = coeff[-nlayers:]
    z = np.linspace(0, np.sum(thickness), 3001) # less than 1um resolution
    if nlayers == 1:
        mu = np.exp(coeff[0]*np.log(energy_list) + coeff[1]).reshape(-1,1)
    elif nlayers == 3:
        mu = np.zeros((energy_list.size, 3))
        mu[:,0] = np.exp(coeff[0]*np.log(energy_list) + coeff[1])
        heaviside = np.zeros(energy_list.size)
        heaviside[energy_bin_ind[1]:] = 1
        mu[:,1] = np.exp(coeff[2]*np.log(energy_list) + coeff[3] + coeff[4]*heaviside)
        heaviside = np.zeros(energy_list.size)
        heaviside[energy_bin_ind[2]:] = 1
        mu[:,2] = np.exp(coeff[5]*np.log(energy_list) + coeff[6] + coeff[7]*heaviside)
    
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
    
    A += 1e-8
    
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

def accuracy_cost_fct(coeff, energy_list):
    thickness = coeff[-nlayers:]
    z = np.linspace(0, np.sum(thickness), 3001) # less than 1um resolution
    if nlayers == 1:
        mu = np.exp(coeff[0]*np.log(energy_list) + coeff[1]).reshape(-1,1)
    elif nlayers == 3:
        mu = np.zeros((energy_list.size, 3))
        mu[:,0] = np.exp(coeff[0]*np.log(energy_list) + coeff[1])
        heaviside = np.zeros(energy_list.size)
        heaviside[energy_bin_ind[1]:] = 1
        mu[:,1] = np.exp(coeff[2]*np.log(energy_list) + coeff[3] + coeff[4]*heaviside)
        heaviside = np.zeros(energy_list.size)
        heaviside[energy_bin_ind[2]:] = 1
        mu[:,2] = np.exp(coeff[5]*np.log(energy_list) + coeff[6] + coeff[7]*heaviside)
    
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
    
    A += 1e-8
    
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

@jit(nopython=True, cache=True)
def accuracy_cost_fct_jit2(coeff, thickness, energy_list, massAttCoeff_element, output=False):
    coeff[1:] /= np.sum(coeff[1:])
    z = np.linspace(0, np.sum(thickness), 3001) # less than 1um resolution
    if nlayers == 1:
        mu = np.exp(np.log(coeff[0]) + np.log(massAttCoeff_element @ coeff[1:].reshape(-1,1))).reshape(-1,1)
    
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
    
    A += 1e-8
    
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

def accuracy_cost_fct2(coeff, thickness, energy_list, massAttCoeff_element):
    coeff[1:] /= np.sum(coeff[1:])
    z = np.linspace(0, np.sum(thickness), 3001) # less than 1um resolution
    if nlayers == 1:
        mu = np.exp(np.log(coeff[0]) + np.log(massAttCoeff_element @ coeff[1:].reshape(-1,1))).reshape(-1,1)
    
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
    
    A += 1e-8
    
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

nElements = 22
energy_bin_ind = np.array([0,17,34,52]).astype(int)
cost_all = np.zeros(1000)
thickness = np.array([0.03]) #cm

#coeff_all = np.zeros((nElements+1, 1000))
#element_list = ['O','F','Na','Al','Si','S','Cl','Ca','Zn','Ge','Se','Br','Y','Cd','I','Cs','Ba','La','Gd','Lu','W','Bi']
#massAttCoeff_element = np.zeros((52, nElements))
#for ne in range(nElements):
#    raw = txt.read_txt(directory + '/data/massAttCoeff' + element_list[ne])
#    massAttCoeff_element[:,ne] = np.interp(energy_list, raw[:,0], raw[:,1], left=0, right=0)
#
#for i in range(1000):
#    print(i)
#    coeff0 = np.random.rand(nElements+1)
#    coeff0[0] = 18*coeff0[0] + 1
#    lb = np.zeros(nElements+1)
#    lb[0] = 1
#    ub = np.ones(nElements+1)
#    ub[0] = 18
#    bnd = Bounds(lb=lb, ub=ub)
#    result = minimize(accuracy_cost_fct_jit2, coeff0, args=(thickness, energy_list, massAttCoeff_element), jac='3-point', method='L-BFGS-B', bounds=bnd)
#    
#    cost_all[i] = result.fun
#    coeff_all[:,i] = result.x
#ind_best = np.argmin(cost_all)
#z, mu, A, PzE, PEz, P, M, acc = accuracy_cost_fct2(coeff_all[:,ind_best], thickness, energy_list, massAttCoeff_element)
##z2, mu2, A2, PzE2, PEz2, P2, M2, acc2 = accuracy_cost_fct(np.array([0.00538629,0.010233,0.0304055]), mu)
#print('Optimization Complete', flush=True)
#np.savez(directory + "/data/optimize_composition_single_layer_67keV_d3_0mm_accuracy_cost", z=z, mu=mu, A=A, PzE=PzE, PEz=PEz, P=P, M=M, acc=acc, thickness=thickness, coeff=coeff_all[:,ind_best],
#         massAttCoeff_element=massAttCoeff_element)#,
##         z2=z2, mu2=mu2, A2=A2, PzE2=PzE2, PEz2=PEz2, P2=P2, M2=M2, acc2=acc2)

#layer_thickness = 0.45
#coeff_all = np.zeros((3, 1000))
#for i in range(1000):
#    print(i)
#    lb = np.array([-2.9,0,layer_thickness])
#    ub = np.array([-2.5,16,layer_thickness])
#    coeff0 = np.random.rand(3)*(ub - lb) + lb
#    bnd = Bounds(lb=lb, ub=ub)
#    result = minimize(accuracy_cost_fct_jit, coeff0, args=(energy_list), jac='3-point', method='L-BFGS-B', bounds=bnd)
#    
#    cost_all[i] = result.fun
#    coeff_all[:,i] = result.x
#ind_best = np.argmin(cost_all)
#z, mu, A, PzE, PEz, P, M, acc = accuracy_cost_fct(coeff_all[:,ind_best], energy_list)
##z2, mu2, A2, PzE2, PEz2, P2, M2, acc2 = accuracy_cost_fct(np.array([0.00538629,0.010233,0.0304055]), mu)
#print('Optimization Complete', flush=True)
#np.savez(directory + "/data/optimize_linAttCoeff_single_layer_67keV_d4_5mm_accuracy_cost", z=z, mu=mu, A=A, PzE=PzE, PEz=PEz, P=P, M=M, acc=acc, thickness=thickness, coeff=coeff_all[:,ind_best])#,
##         z2=z2, mu2=mu2, A2=A2, PzE2=PzE2, PEz2=PEz2, P2=P2, M2=M2, acc2=acc2)

layer_thickness = 0.15
coeff_all = np.zeros((11, 1000))
for i in range(1000):
    print(i)
    lb = np.array([-2.9,0,-2.9,0,0,-2.9,0,0,layer_thickness,layer_thickness,layer_thickness])
    ub = np.array([-2.5,16,-2.5,16,1.9,-2.5,16,1.9,layer_thickness,layer_thickness,layer_thickness])
    coeff0 = np.random.rand(11)*(ub - lb) + lb
    bnd = Bounds(lb=lb, ub=ub)
    result = minimize(accuracy_cost_fct_jit, coeff0, args=(energy_list), jac='3-point', method='L-BFGS-B', bounds=bnd)
    
    cost_all[i] = result.fun
    coeff_all[:,i] = result.x
ind_best = np.argmin(cost_all)
z, mu, A, PzE, PEz, P, M, acc = accuracy_cost_fct(coeff_all[:,ind_best], energy_list)
#z2, mu2, A2, PzE2, PEz2, P2, M2, acc2 = accuracy_cost_fct(np.array([0.00538629,0.010233,0.0304055]), mu)
print('Optimization Complete', flush=True)
np.savez(directory + "/data/optimize_linAttCoeff_three_layer_123config_67keV_d4_5mm_accuracy_cost", z=z, mu=mu, A=A, PzE=PzE, PEz=PEz, P=P, M=M, acc=acc, thickness=thickness, coeff=coeff_all[:,ind_best])#,
#         z2=z2, mu2=mu2, A2=A2, PzE2=PzE2, PEz2=PEz2, P2=P2, M2=M2, acc2=acc2)

#coeff0 = np.linspace(-2.9, -2.5, 101)
#coeff1 = np.linspace(6, 16, 101)
#acc_all = np.zeros((101, 101))
#for i in range(101):
#    for j in range(101):
#        coeff_temp = np.array([coeff0[i],coeff1[j]])
#        acc_all[i,j] = accuracy_cost_fct_jit(coeff_temp, thickness, energy_list)
#ind_best0 = np.where(acc_all==np.nanmin(acc_all))[0][0]
#ind_best1 = np.where(acc_all==np.nanmin(acc_all))[1][0]
#z, mu, A, PzE, PEz, P, M, acc = accuracy_cost_fct(np.array([coeff0[ind_best0],coeff1[ind_best1]]), thickness, energy_list)
#print('Sweep Complete', flush=True)
#np.savez(directory + "/data/sweep_linAttCoeff_single_layer_67keV_d3_0mm_accuracy_cost", z=z, mu=mu, A=A, PzE=PzE, PEz=PEz, P=P, M=M, acc=acc, thickness=thickness,
#         acc_all=acc_all, coeff0=coeff0, coeff1=coeff1)