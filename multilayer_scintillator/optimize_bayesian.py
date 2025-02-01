import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
from scipy.linalg import svd
import subprocess
import getpass
import uproot
import time
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import UtilityFunction
from itertools import permutations
import multicolor_scintillator.merit_fct as mf

class optimizer:
    def __init__(self, simulation, d_bounds, n_init, n_max, Nscint, starting_permID, load_data):
        self.simulation = simulation
        self.d_bounds = d_bounds
        self.n_init = n_init
        self.n_max = n_max
        self.Nscint = Nscint
        self.starting_permID = starting_permID
        self.load_data = load_data
        
    def four_layer_fom(self, d1, d2, d3):
        fom = self.simulation.multicolor_scintillator_FoM(thickness=np.array([d1,d2,d3])) # in mm
        
        return fom
    
    def run_bayesian(self):
        obj = BayesianOptimization(self.four_layer_fom, self.d_bounds)
                                   #bounds_transformer=SequentialDomainReductionTransformer(minimum_window=0.5))
        acq = UtilityFunction(kind='ucb', kappa=10)
        obj.set_gp_params(alpha=1e-3)
        obj.maximize(init_points=self.n_init, n_iter=self.n_max) #, acquisition_function=acq)
        d1 = obj.max['params']['d1']
        d2 = obj.max['params']['d2']
        d3 = obj.max['params']['d3']
        
        d_iter = np.zeros((self.n_max + self.n_init, self.Nscint))
        fom_iter = np.zeros(self.n_max + self.n_init)
        for i, res in enumerate(obj.res):
            d_iter[i,0] = res['params']['d1']
            d_iter[i,1] = res['params']['d2']
            d_iter[i,2] = res['params']['d3']
            fom_iter[i] = res['target']
        
        return np.array([d1,d2,d3]), d_iter, fom_iter
    
    def optimize_all_permutations(self):
        scintillator_permutations = np.array(list(permutations(np.arange(self.Nscint).astype(int) + 1)))
        if self.load_data:
            with np.load(directory + "/results/optimization_result.npz") as data:
                optimal_thicknesses = data['optimal_thicknesses']
                d_iter_all = data['d_iter_all']
                fom_iter_all = data['fom_iter_all']
        else:
            optimal_thicknesses = np.zeros_like(scintillator_permutations).astype(np.float64)
            d_iter_all = np.zeros((scintillator_permutations.shape[0], self.n_max + self.n_init, self.Nscint))
            fom_iter_all = np.zeros((scintillator_permutations.shape[0], self.n_max + self.n_init))
            
        for i in range(self.starting_permID, scintillator_permutations.shape[0]):
            print("\n### Permutation %d: %d %d %d\n" %(i, scintillator_permutations[i,0],
                                                       scintillator_permutations[i,1],
                                                       scintillator_permutations[i,2]), flush=True)
            self.simulation.permutationID = i
            self.simulation.scintillator_material = scintillator_permutations[i,:]
            optimal_thicknesses[i,:], d_iter_all[i,:,:], fom_iter_all[i,:] = self.run_bayesian()
            
            self.simulation.Nphoton *= 100
            FoM = self.simulation.multicolor_scintillator_FoM(thickness=optimal_thicknesses[i,:])
            # self.simulation.analyze_all_energies(optimal_thicknesses[i,:])
            self.simulation.Nphoton /= 100
            
            np.savez(directory + "/results/optimization_result", scintillator_permutations=scintillator_permutations,
                     optimal_thicknesses=optimal_thicknesses, d_iter_all=d_iter_all,
                     fom_iter_all=fom_iter_all)

if __name__ == '__main__':
    simulation = mf.geant4(G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main",
                           dimScint = np.array([1,1]), # in cm
                           dimDet = np.array([1,1]), # in cm
                           gap = 10, # in cm
                           structureID = 1,
                           Nlayers = 3,
                           scintillator_material = np.array([1,2,3]),
                           max_thickness = 1, # in mm
                           Ngrid = np.array([1,1]),
                           detector_thickness = 0.1, # in um
                           check_overlap = 1,
                           # Nthreads = 8,
                           energy_range = np.array([10,100]), # in keV
                           energy_bins = 10,
                           Nphoton = 10000,
                           wvl_bins = 801,
                           calibration_method='unity')
    d_bounds = {'d1': (0.1, 1), 'd2': (0.1, 1), 'd3': (0.1, 1)}
    wrapper = optimizer(simulation, d_bounds, n_init=10, n_max=90, Nscint=3, starting_permID=2, load_data=False)
    wrapper.optimize_all_permutations()