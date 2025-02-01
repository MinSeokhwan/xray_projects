import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
import multilayer_scintillator.pso_merit_fct as mf
import matplotlib.pyplot as plt
import time

from mpi4py import MPI
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

def PSO(identifier, mode, swarm_size, var_range, iteration_limit, stop_limit=100, c1=3, c2=2, w=0.8, starting_iteration=1, stop_count=0, simulation=None, error_tolerance=False):
    """ mode: searching for 'min' or 'max'
        swarm_size: should be divisible by 10
        var_range: (dimension x 2 -> min, max)
        merit_fct: a python def (put no brackets at the end) (set default values for all fixed arguments)
        iteration_limit: max number of total iterations
        stop_limit: max number of iterations without improvement """
    
    iteration_count = starting_iteration
    dimension = np.shape(var_range)[0]
    vmax = 0.4*np.abs(var_range[:,1] - var_range[:,0])
    v = np.zeros((dimension, swarm_size))
    index = np.zeros((dimension, swarm_size))
    
    if rank == 0:
        #Particle Initialization
        if starting_iteration > 1:
            index = np.load(directory + '/data/particle_index_' + identifier + '.npy')
            v = np.load(directory + '/data/particle_v_' + identifier + '.npy')
            
            pbest_val = np.load(directory + '/data/pbest_val_' + identifier + '.npy')
            pbest_ind = np.load(directory + '/data/pbest_ind_' + identifier + '.npy')
            
            gbest_val = np.load(directory + '/data/gbest_val_' + identifier + '.npy')
            gbest_ind = np.load(directory + '/data/gbest_ind_' + identifier + '.npy')
        else:
            #Latin Hypercube Method
            increment = (var_range[:,1] - var_range[:,0])/swarm_size
            for dim in range(dimension):
                swarm_ind = np.random.permutation(swarm_size).reshape((1, swarm_size))
                if var_range[dim,0] == var_range[dim,1]:
                    index[dim,:] = var_range[dim,0]*np.ones((1, swarm_size))
                    v[dim,:] = np.zeros((1, swarm_size))
                else:
                    index[dim,:] = var_range[dim,0] + (swarm_ind+1)*increment[dim] - increment[dim]*np.random.rand(1, swarm_size)
                    v[dim,:] = -vmax[dim] + 2*vmax[dim]*np.random.rand(1, swarm_size)
    
            pbest_val = np.zeros((1, swarm_size))
            pbest_ind = np.zeros((dimension, swarm_size))
            
            gbest_val = 0
            gbest_ind = np.zeros((dimension, 1))
    
        if error_tolerance:
            # For Error-Tolerance Computation
            stop_limit = 1
            index[:,:] = np.array([0.527273,0.809091,0.184700])[:,np.newaxis]
            index[0,:] = np.linspace(var_range[0,0], var_range[0,1], swarm_size)
            print(index)

    #Text File for Evaluation Status
    with open(directory + '/logs/PSO_status_' + identifier + '.txt', 'w') as f:
        f.write('-----\n')
    
    quo, rem = divmod(swarm_size, size)
    data_size = np.array([quo + 1 if p < rem else quo for p in range(size)]).astype(np.int32)
    data_disp = np.array([sum(data_size[:p]) for p in range(size)]).astype(np.int32)
    while True:
        if iteration_count > iteration_limit or stop_count >= stop_limit:
            # simulation.Nphoton *= 10
            if rank == 0:
                gbest_thickness = np.ascontiguousarray(gbest_ind[:,-1])
            else:
                gbest_thickness = np.zeros(dimension, dtype=np.float64)
            comm.Bcast(gbest_thickness, root=0)
            
            FoM = simulation.FoM(0, thickness=gbest_thickness)
            break
        
        # Particle Statistics
        if rank == 0:
            for dim in range(dimension):
                fig, ax = plt.subplots(dpi=100)
                ax.hist(index[dim,:], bins=100, density=True)
                plt.savefig(directory + '/plots/PSO_stats_index_' + str(dim) + '.png', dpi=100)
                plt.close()
        
        with open(directory + '/logs/PSO_status_' + identifier + '.txt', 'a') as f:
            f.write('Iteration %d\n' %iteration_count)
        
        index_proc = np.zeros((dimension, data_size[rank]))
        for dim in range(dimension):
            if rank == 0:
                index_temp = index[dim,:]
            else:
                index_temp = None
            comm.Scatterv([index_temp, data_size, data_disp, MPI.DOUBLE], index_proc[dim,:], root=0)
        
        valF_proc = np.zeros(data_size[rank])
#        if rank == 1:
#            print('index_1: ' + str(index_proc))
        for s in range(data_size[rank]):
            valF_proc[s] = simulation.FoM(rank, index_proc[:,s])

        valF = np.zeros(swarm_size)
        comm.Gatherv(valF_proc, [valF, data_size, data_disp, MPI.DOUBLE], root=0)

        if rank == 0:
            np.save(directory + '/data/valF_' + identifier + '.npy', valF)
        
            #Particle Best
            for s in range(swarm_size):
                if iteration_count == 1:
                    pbest_val[0,s] = valF[s]
                    pbest_ind[:,s] = index[:,s]
                elif mode == 'min':
                    if valF[s] < pbest_val[0,s]:
                        pbest_val[0,s] = valF[s]
                        pbest_ind[:,s] = index[:,s]
                elif mode == 'max':
                    if valF[s] > pbest_val[0,s]:
                        pbest_val[0,s] = valF[s]
                        pbest_ind[:,s] = index[:,s]
                
            #Global Best
            if mode == 'min':
                new_Gbest = np.min(pbest_val)
                new_Gbest_pos = np.argmin(pbest_val)
            elif mode == 'max':
                new_Gbest = np.max(pbest_val)
                new_Gbest_pos = np.argmax(pbest_val)
            new_Gbest_ind = pbest_ind[:,new_Gbest_pos].reshape(dimension,1)
            
            if iteration_count == 1:
                gbest_val = new_Gbest
                gbest_ind = new_Gbest_ind
            elif iteration_count == 2:
                if new_Gbest == gbest_val:
                    stop_count += 1
                    gbest_val = np.append(gbest_val, new_Gbest)
                    gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                elif mode == 'min':
                    if new_Gbest < gbest_val:
                        stop_count = 1
                        gbest_val = np.append(gbest_val, new_Gbest)
                        gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                elif mode == 'max':
                    if new_Gbest > gbest_val:
                        stop_count = 1
                        gbest_val = np.append(gbest_val, new_Gbest)
                        gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
            else:
                if new_Gbest == gbest_val[-1]:
                    stop_count += 1
                    gbest_val = np.append(gbest_val, new_Gbest)
                    gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                elif mode == 'min':
                    if new_Gbest < gbest_val[-1]:
                        stop_count = 1
                        gbest_val = np.append(gbest_val, new_Gbest)
                        gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
                elif mode == 'max':
                    if new_Gbest > gbest_val[-1]:
                        stop_count = 1
                        gbest_val = np.append(gbest_val, new_Gbest)
                        gbest_ind = np.hstack((gbest_ind, new_Gbest_ind))
            with open(directory + '/logs/PSO_status_' + identifier + '.txt', 'a') as f:
                f.write('Global Best Index:\t')
                for dim in range(dimension):
                    f.write('%f\t' %gbest_ind[dim,-1])
                if np.size(gbest_val) == 1:
                    f.write('\nGlobal Best Value: %f\n\n' %gbest_val)
                else:
                    f.write('\nGlobal Best Value: %f\n\n' %gbest_val[-1])
                
            #Parameter Updates
            if stop_count >= 5:
                w = w*0.99
                vmax = vmax*0.99
            
            #Velocity Updates
            for s in range(swarm_size):
                v[:,s] = w*v[:,s] + c1*np.random.rand()*(pbest_ind[:,s] - index[:,s]) + c2*np.random.rand()*(gbest_ind[:,-1] - index[:,s])
                for dim in range(dimension):
                    if np.abs(v[dim,s]) > vmax[dim]:
                        v[:,s] = v[:,s]*vmax[dim]/np.abs(v[dim,s]) #reduce magnitude while maintaining direction
            
            #Craziness
            N_cr = np.random.permutation(swarm_size)[:int(swarm_size/10)]
            if np.random.rand() < 0.22:
                for t in range(int(swarm_size/10)):
                    v[:,N_cr[t]] = -vmax + vmax*np.random.rand(dimension)
            
            #Position Updates
            for s in range(swarm_size):
                for dim in range(dimension):
                    index[dim,s] += v[dim,s]
                    if index[dim,s] < var_range[dim,0]:
                        index[dim,s] = var_range[dim,0] + (var_range[dim,0] - index[dim,s])
                    elif index[dim,s] > var_range[dim,1]:
                        index[dim,s] = var_range[dim,1] - (index[dim,s] - var_range[dim,1])
            
            #Write Results into Text File (optional)
            if iteration_count == 1:
                with open(directory + '/logs/PSO_global_' + identifier + '.txt', 'w') as f:
                    f.write('Iteration %d:\t' % iteration_count)
                    f.write(np.format_float_scientific(gbest_val))
                    f.write('\t\t')
                    for dim in range(dimension):
                        #f.write(np.format_float_scientific(gbest_ind[dim,-1]))
                        f.write('%f\t' % gbest_ind[dim,-1])
                    f.write('\n')
            else:
                if os.path.exists(directory + '/logs/PSO_global_' + identifier + '.txt'):
                    with open(directory + '/logs/PSO_global_' + identifier + '.txt', 'a') as f:
                        f.write('Iteration %d:\t' % iteration_count)
                        f.write(np.format_float_scientific(gbest_val[-1]))
                        f.write('\t\t')
                        for dim in range(dimension):
                            #f.write(np.format_float_scientific(gbest_ind[dim,-1]))
                            f.write('%f\t' % gbest_ind[dim,-1])
                        f.write('\n')
                else:
                    with open(directory + '/logs/PSO_global_' + identifier + '.txt', 'w') as f:
                        f.write('Iteration %d:\t' % iteration_count)
                        f.write(np.format_float_scientific(gbest_val[-1]))
                        f.write('\t\t')
                        for dim in range(dimension):
                            #f.write(np.format_float_scientific(gbest_ind[dim,-1]))
                            f.write('%f\t' % gbest_ind[dim,-1])
                        f.write('\n')
            
            #Save Progress
            np.save(directory + '/data/particle_index_' + identifier + '.npy', index)
            np.save(directory + '/data/particle_v_' + identifier + '.npy', v)
            np.save(directory + '/data/pbest_val_' + identifier + '.npy', pbest_val)
            np.save(directory + '/data/pbest_ind_' + identifier + '.npy', pbest_ind)
            np.save(directory + '/data/gbest_val_' + identifier + '.npy', gbest_val)
            np.save(directory + '/data/gbest_ind_' + identifier + '.npy', gbest_ind)
        
        if rank != 0:
            reqs = comm.isend(rank, dest=0)
            reqs.wait()
        else:
            with open(directory + '/logs/PSO_status_' + identifier + '.txt', 'a') as f:
                f.write('Checkpoint:\t')
                for i in range(1,size):
                    reqr = comm.irecv(source=i)
                    dummy = reqr.wait()
                    f.write('%d\t' %dummy)
                f.write('\n\n')
        
        iteration_count += 1
        stop_count = comm.bcast(stop_count, root=0)

if __name__ == '__main__':
#    np.random.seed(0)
#    Ntrain = 100
#    training_data = np.zeros((Ntrain, 3))
#    training_data[:,:2] = 2*(np.random.rand(Ntrain, 2) - 1)
#    training_data[:,2] = 3*(np.random.rand(Ntrain) - 1) - 3
#    training_data = 10**training_data
#    training_data *= 2/np.sum(training_data, axis=1)[:,np.newaxis]

    if rank == 0:
        verbose = True
    else:
        verbose = False

    identifier = 'energy_bins_30_35_47_52keV_245' #'energy_bins_30_35_40keV'
    simulation = mf.geant4(G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main",
                           dimScint = np.array([1.5,1.5]), # in cm
                           dimDetVoxel = np.array([12800,12800,0.1]), # in um
                           NDetVoxel = np.array([1,1,1]),
                           gapSampleScint = 0.0, # in cm
                           gapScintDet = 0.0, # in cm
                           Nlayers = 3,
                           scintillator_material = np.array([2,4,5]), # 1:YAGCe, 2:ZnSeTe, 3:LYSOCe, 4:CsITl, 5:GSOCe
                           check_overlap = 1,
                           energy_range = None, # np.array([30,120]), # in keV
                           energy_bins = None, # 3,
                           energy_list = np.array([30.5,35.5,47.5,52.5]), # in keV
                           Nphoton = 10000,
                           wvl_bins = 161, # 300 nm ~ 1100 nm
                           source_type = 'uniform', # uniform / xray_tube
                           xray_target_material = 'W',
                           xray_operating_V = 100, # in keV
                           xray_tilt_theta = 12, # in deg
                           xray_filter_material = 'Al',
                           xray_filter_thickness = 0.3, # in cm
                           FoM_type = 'N_energy_bins', # K_edge_imaging_v2 / K_edge_imaging / N_energy_bins
                           verbose = verbose,
                           identifier = identifier,
                           )
    
    if rank == 0:
        simulation.cmake()
        simulation.make()
    else:
        time.sleep(60)
    
    # FoM = simulation.FoM(0, thickness=np.array([0.5,0.5,0.5]), gamma_optimization=True)
    
#    gamma_data = np.load(directory + '/results/reg_matrix1_init_' + simulation.identifier + '.npz')
#    simulation.gamma_block_opt = gamma_data['gamma_block']
    
    var_range = np.array([[0.01,0.5],
                          [0.01,0.5],
                          [0.01,1]])
    
    PSO(identifier = identifier,
        mode = "min",
        swarm_size = 1000,
        var_range = var_range,
        iteration_limit = 100,
        stop_limit = 10,
        c1 = 1.5,
        c2 = 1.5,
        w = 0.9,
        starting_iteration = 1,
        stop_count = 0,
        simulation = simulation,
        error_tolerance = False,
        )