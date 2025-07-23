import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-31])

import numpy as np
from numba import jit
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# incident angle vs. E_X_det and n_scint_img and n_scint_corr E_X_corr

class postprocessor:
    def __init__(self, identifier):
        self.identifier = identifier

    def collect_angle_dependent_data(self):
        cnt = 0
        theta_inc_list = np.zeros(0)
        E_X_corr = None
        E_X_det = None
        n_scint_corr = None
        n_scint_det = None
        while True:
            print(cnt, flush=True)
            filename = directory + '/data/' + self.identifier + '/geant4_data_' + str(cnt) + '.npz'
            if not os.path.exists(filename):
                break
            
            with np.load(filename) as raw_data:
                theta_inc_list_temp, E_X_corr_temp = self.get_stats('E_X_corr', raw_data)
                theta_inc_list = np.hstack((theta_inc_list, theta_inc_list_temp))
                if E_X_corr is None:
                    E_X_corr = E_X_corr_temp.copy()
                else:
                    E_X_corr = np.vstack((E_X_corr, E_X_corr_temp))

                theta_inc_list_temp, E_X_det_temp = self.get_stats('E_X_det', raw_data)
                if E_X_det is None:
                    E_X_det = E_X_det_temp.copy()
                else:
                    E_X_det = np.vstack((E_X_det, E_X_det_temp))
                
                theta_inc_list_temp, n_scint_corr_temp = self.get_stats('n_scint_corr', raw_data)
                if n_scint_corr is None:
                    n_scint_corr = n_scint_corr_temp.copy()
                else:
                    n_scint_corr = np.vstack((n_scint_corr, n_scint_corr_temp))

                theta_inc_list_temp, n_scint_det_temp = self.get_stats('n_scint_det', raw_data)
                if n_scint_det is None:
                    n_scint_det = n_scint_det_temp.copy()
                else:
                    n_scint_det = np.vstack((n_scint_det, n_scint_det_temp))
            
            cnt += 1
        
        np.savez(directory + '/data/' + self.identifier + '/postprocessed_data',
            theta_inc_list=theta_inc_list,
            E_X_corr=E_X_corr,
            E_X_det=E_X_det,
            n_scint_corr=n_scint_corr,
            n_scint_det=n_scint_det
        )

    def get_stats(self, quantity, raw_data):
        imaging_mask = raw_data['n_scint_det'] > 0
        theta_inc_list = np.unique(raw_data['theta_inc'])
        theta_inc_all = raw_data['theta_inc'][imaging_mask]
        filtered_data = raw_data[quantity][imaging_mask]
        print(theta_inc_list*180/np.pi)

        theta_dependent_data = np.zeros((theta_inc_list.size, 4))
        for i in range(theta_inc_list.size):
            theta_mask = theta_inc_all == theta_inc_list[i]
            theta_dependent_data[i,0] = np.min(filtered_data[theta_mask])
            theta_dependent_data[i,1] = np.mean(filtered_data[theta_mask])
            theta_dependent_data[i,2] = np.max(filtered_data[theta_mask])
            theta_dependent_data[i,3] = np.sum(filtered_data[theta_mask])
        
        return theta_inc_list, theta_dependent_data

if __name__ == '__main__':
    identifier = 'YAGCe_50_120keV_tube_Pb3mm_calibration'
    
    postproc = postprocessor(identifier)
    postproc.collect_angle_dependent_data()

    print('Done', flush=True)