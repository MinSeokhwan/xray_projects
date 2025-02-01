import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-28])

import numpy as np
# import scipy.special
# import scipy.ndimage
import autograd
from autograd import numpy as npa, grad
from autograd.extend import primitive, defvjp
import pylops
import time
from mpi4py import MPI

from multilayer_scintillator import lasso

comm = MPI.COMM_WORLD

###################################################
## Reconstruction code building on Gaurav's code ##
###################################################

make_pyLop = lambda X: pylops.MatrixMult(X)
make_matvec = lambda X: (lambda u: X @ u)
make_matvec_params = lambda params: make_matvec(params[0])
make_matTvec = lambda X: (lambda y: X.T @ y)
make_matTvec_params = lambda params: make_matTvec(params[0])
make_pyLop_params = lambda params: make_pyLop(params[0])

iter_number = 1
make_alpha = lambda params: 10 ** params[1]
make_beta = lambda params: 10 ** params[2]

##############################################################
## Outputs Reconstruction Results (for forward computation) ##
##############################################################

def reconstruct(params_new, training_data, noise_function):
    Gu = np.matmul(params_new[0], training_data)

    uest = lasso.solve(params_new,
                       make_pyLop_params,
                       make_matvec_params,
                       make_matTvec_params,
                       noise_function,
                       training_data,
                       make_alpha,
                       make_beta,
                       max_iters=2000000,
                       tol=1e-12,
                       verbosity=0,
                       restart=-1)
    sys.stdout.flush()
    
    return uest

###############################################
## Outputs Reconstruction Results (autograd) ##
###############################################

def find_loss_from_Wd(params_new, training_data, noise_function, iter_number):
    loss_vec = loss_from_Wd(params_new, training_data, noise_function, iter_number)
    return 1e3*loss_vec[0]/loss_vec[1]

@primitive
def loss_from_Wd(params_new, training_data, noise_function, iter_number):
    Gu = np.matmul(params_new[0], training_data)
    
    if mp.am_really_master():
        verbosity = 0
    else:
        verbosity = 0
    
    uest = lasso.solve(params_new,
                       make_pyLop_params,
                       make_matvec_params,
                       make_matTvec_params,
                       noise_function,
                       training_data,
                       make_alpha,
                       make_beta,
                       max_iters=2000000,
                       tol=1e-12,
                       verbosity=verbosity,
                       restart=-1)
                       
    ret_numerator = np.sum((training_data - uest) ** 2)
    ret_denominator = np.sum(training_data**2)
    sys.stdout.flush()
    
    return np.array([ret_numerator,ret_denominator])

def reconstruction_hack(vec, params_new, training_data, noise_function, iter_number):
    def proxy(params_new):
        if mp.am_really_master():
            verbosity = 0
        else:
            verbosity = 0
    
        uest = lasso.solve(params_new,
                           make_pyLop_params,
                           make_matvec_params,
                           make_matTvec_params,
                           noise_function,
                           training_data,
                           make_alpha,
                           make_beta,
                           max_iters=2000000,
                           tol=1e-12,
                           verbosity=verbosity,
                           restart=-1)
                           
        ret_numerator = npa.sum((training_data - uest) ** 2)
        ret_denominator = npa.sum(training_data**2)
        sys.stdout.flush()
        
        process_result = npa.array([ret_numerator,ret_denominator])
        return npa.dot(vec, process_result)

    g = grad(proxy)
    return g(params_new)

def backward_loss_from_Wd(obj, params_new, training_data, noise_function, iter_number):
    def vjp(vec):
        grad_tuple = reconstruction_hack(vec, params_new, training_data, noise_function, iter_number)
        return grad_tuple

    return vjp

defvjp(loss_from_Wd, backward_loss_from_Wd, argnums=[0])

#################################################
## Parallel Computation of All Training Losses ##
#################################################

@primitive
def training_loss_from_Wd(grad_params, device):
    if device.robust_type == 'original':
        Wd = device.Wd_iter0.reshape(device.Npixels[0]*device.Npixels[1], device.Nfreq*device.Ntheta*device.Nphi)
    elif device.robust_type == 'eroded':
        Wd = device.Wd_iter_erosion.reshape(device.Npixels[0]*device.Npixels[1], device.Nfreq*device.Ntheta*device.Nphi)
    elif device.robust_type == 'dilated':
        Wd = device.Wd_iter_dilation.reshape(device.Npixels[0]*device.Npixels[1], device.Nfreq*device.Ntheta*device.Nphi)
    reconstr_alpha = grad_params[0]
    reconstr_beta = grad_params[1]

    # Divide Data Between Processes
    Nproc = int(np.min((comm.size, device.Nbatch)))
    quo, rem = divmod(device.Nbatch, Nproc)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(Nproc)]).astype(np.int32)
    data_disp = np.array([sum(data_size[:p]) for p in range(Nproc+1)]).astype(np.int32)
    training_data_per_proc = device.training_data[data_disp[comm.rank]:data_disp[comm.rank+1],:]

    # Define Noise Function
    if device.apply_noise:
        noise_fct = device.noise_function
    else:
        noise_fct = lambda _: 0
    
    # Run Reconstruction Algorithm
    params = autograd.builtins.tuple((Wd, reconstr_alpha, reconstr_beta, device.reconstr_noise_sigma))
    loss_proc = np.zeros(data_size[comm.rank])
    reconstr_proc = np.zeros_like(training_data_per_proc)
    
    for n_data in range(data_size[comm.rank]):
        loss_proc[n_data] = find_loss_from_Wd(params, training_data_per_proc[n_data,:], noise_fct, device.n_iter)
        reconstr_proc[n_data,:] = reconstruct(params, training_data_per_proc[n_data,:], noise_fct)

    # Allgather Loss and Reconstruction Results
    loss = np.zeros(device.Nbatch)
    comm.Allgatherv(loss_proc, [loss, data_size, data_disp[:-1], MPI.DOUBLE])
    
    data_size_temp = data_size*device.Nfreq*device.Ntheta*device.Nphi
    data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)]).astype(np.float64)
    
    reconstr_all_temp = np.zeros(device.Nbatch*device.Nfreq*device.Ntheta*device.Nphi)
    comm.Allgatherv(reconstr_proc.reshape(-1), [reconstr_all_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
    reconstr_all = reconstr_all_temp.reshape(device.Nbatch, device.Nfreq*device.Ntheta*device.Nphi)
    
    if device.robust_type == 'original':
        device.loss_all0 = loss
        device.reconstr_all0 = reconstr_all
    elif device.robust_type == 'eroded':
        device.loss_all_erosion = loss
        device.reconstr_all_erosion = reconstr_all
    elif device.robust_type == 'dilated':
        device.loss_all_dilation = loss
        device.reconstr_all_dilation = reconstr_all
    
    return loss

def vjp_training_loss_from_Wd(loss, grad_params, device):
    if device.robust_type == 'original':
        Wd = device.Wd_iter0.reshape(device.Npixels[0]*device.Npixels[1], device.Nfreq*device.Ntheta*device.Nphi)
    elif device.robust_type == 'eroded':
        Wd = device.Wd_iter_erosion.reshape(device.Npixels[0]*device.Npixels[1], device.Nfreq*device.Ntheta*device.Nphi)
    elif device.robust_type == 'dilated':
        Wd = device.Wd_iter_dilation.reshape(device.Npixels[0]*device.Npixels[1], device.Nfreq*device.Ntheta*device.Nphi)
    reconstr_alpha = grad_params[0]
    reconstr_beta = grad_params[1]

    def vjp(g):
        # Divide Data Between Processes
        Nproc = int(np.min((comm.size, device.Nbatch)))
        quo, rem = divmod(device.Nbatch, Nproc)
        data_size = np.array([quo + 1 if p < rem else quo for p in range(Nproc)]).astype(np.int32)
        data_disp = np.array([sum(data_size[:p]) for p in range(Nproc+1)]).astype(np.int32)
        training_data_per_proc = device.training_data[data_disp[comm.rank]:data_disp[comm.rank+1],:]
        
        # Define Noise Function
        if device.apply_noise:
            noise_fct = device.noise_function
        else:
            noise_fct = lambda _: 0
        
        # Run Reconstruction Algorithm
        params = autograd.builtins.tuple((Wd, reconstr_alpha, reconstr_beta, device.reconstr_noise_sigma))
        jac_alpha_proc = np.zeros(data_size[comm.rank])
        jac_beta_proc = np.zeros(data_size[comm.rank])
            
        for n_data in range(data_size[comm.rank]):
            jac_temp = grad(find_loss_from_Wd)(params, training_data_per_proc[n_data,:], noise_fct, device.n_iter)
            jac_alpha_proc[n_data] = jac_temp[1]
            jac_beta_proc[n_data] = jac_temp[2]
    
        # Allgather Loss and Reconstruction Results
        jac_alpha = np.zeros(device.Nbatch)
        comm.Allgatherv(jac_alpha_proc, [jac_alpha, data_size, data_disp[:-1], MPI.DOUBLE])
        
        jac_beta = np.zeros(device.Nbatch)
        comm.Allgatherv(jac_beta_proc, [jac_beta, data_size, data_disp[:-1], MPI.DOUBLE])
    
        # Compute VJPs
        jac_alpha = np.dot(g, jac_alpha)
        jac_beta = np.dot(g, jac_beta)
        
        return autograd.builtins.tuple((jac_alpha, jac_beta))
    
    return vjp

defvjp(training_loss_from_Wd, vjp_training_loss_from_Wd, argnums=[0])

defvjp(training_loss_from_fields, vjp_training_loss_from_fields, argnums=[0])