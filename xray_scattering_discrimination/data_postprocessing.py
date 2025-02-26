import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-31])

import numpy as np
import time

def merge_datafiles(identifier, ind_max):
    print('### Merging Data Files', end='', flush=True)
    for i in range(ind_max):
        print(' | ' + str(i), end='', flush=True)
        with np.load(directory + "/data/" + identifier + "/geant4_data_" + str(i) + ".npz") as data:
            if i == 0:
                E_inc = data['E_inc']
                dE_trans = data['dE_trans']
                theta_trans = data['theta_trans']
                depth_scint = data['depth_scint']
                R_scint = data['R_scint']
                dxy_scint = data['dxy_scint']
                n_scint = data['n_scint']
                n_rayl = data['n_rayl']
                n_phot = data['n_phot']
                n_compt = data['n_compt']
            
            else:
                E_inc = np.hstack((data['E_inc'], E_inc))
                dE_trans = np.hstack((data['dE_trans'], dE_trans))
                theta_trans = np.hstack((data['theta_trans'], theta_trans))
                depth_scint = np.hstack((data['depth_scint'], depth_scint))
                R_scint = np.vstack((data['R_scint'], R_scint))
                dxy_scint = np.vstack((data['dxy_scint'], dxy_scint))
                n_scint = np.hstack((data['n_scint'], n_scint))
                n_rayl = np.hstack((data['n_rayl'], n_rayl))
                n_phot = np.hstack((data['n_phot'], n_phot))
                n_compt = np.hstack((data['n_compt'], n_compt))

    print('', flush=True)

    return E_inc, dE_trans, theta_trans, depth_scint, R_scint, dxy_scint, n_scint, n_rayl, n_phot, n_compt

def compute_stats(E_inc,
                  dE_trans,
                  theta_trans,
                  depth_scint,
                  R_scint,
                  dxy_scint,
                  n_scint,
                  n_rayl,
                  n_phot,
                  n_compt,
                  ):

    print('### Computing Relevant Statistics', flush=True)
    max_rayl = int(np.max(n_rayl)) + 1
    max_phot = int(np.max(n_phot)) + 1
    max_compt = int(np.max(n_compt)) + 1
    n_E = int(np.ptp(E_inc)) + 1

    mask_rayl = np.zeros((n_rayl.size, max_rayl))
    for i in range(max_rayl):
        mask_rayl[:,i] = np.abs(n_rayl - i) < 0.1
    
    mask_phot = np.zeros((n_phot.size, max_phot))
    for i in range(max_phot):
        mask_phot[:,i] = np.abs(n_phot - i) < 0.1
    
    mask_compt = np.zeros((n_compt.size, max_compt))
    for i in range(max_compt):
        mask_compt[:,i] = np.abs(n_compt - i) < 0.1
    
    print('\t-Number of occurence', flush=True)
    n_eventType = np.zeros((max_rayl, max_phot, max_compt, n_E))
    for i in range(max_rayl):
        for j in range(max_phot):
            for k in range(max_compt):
                for l in range(n_E):
                    n_eventType[i,j,k,l] = np.sum(mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]*(np.abs(E_inc - (l+np.min(E_inc))) < 0.5))
    
    def collect_stats(raw):
        stat = np.zeros((max_rayl, max_phot, max_compt, 2))
        nanmask = (1 - np.isnan(raw)).astype(bool)
        for i in range(max_rayl):
            for j in range(max_phot):
                for k in range(max_compt):
                    mask = (mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]).astype(bool)
                    
                    raw_masked = raw[mask*nanmask]
                    if raw_masked.size > 0:
                        stat[i,j,k,0] = np.mean(raw[mask*nanmask])
                        stat[i,j,k,1] = np.std(raw[mask*nanmask])
                    else:
                        stat[i,j,k,0] = np.nan
                        stat[i,j,k,1] = np.nan
        
        return stat
    
    def collect_spectral_stats(raw):
        stat = np.zeros((max_rayl, max_phot, max_compt, n_E, 2))
        nanmask = (1 - np.isnan(raw)).astype(bool)
        for i in range(max_rayl):
            for j in range(max_phot):
                for k in range(max_compt):
                    for l in range(n_E):
                        mask = (mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]*(np.abs(E_inc - (l+np.min(E_inc))) < 0.5)).astype(bool)
                        
                        raw_masked = raw[mask*nanmask]
                        if raw_masked.size > 0:
                            stat[i,j,k,l,0] = np.mean(raw[mask*nanmask])
                            stat[i,j,k,l,1] = np.std(raw[mask*nanmask])
                        else:
                            stat[i,j,k,l,0] = np.nan
                            stat[i,j,k,l,1] = np.nan
        
        return stat
    
    print('\t-Incident energy', flush=True)
    stat_E_inc = collect_stats(E_inc)
    
    print('\t-Energy loss', flush=True)
    stat_dE_trans = collect_spectral_stats(dE_trans)
    
    print('\t-X-ray exit angle', flush=True)
    stat_theta_trans = collect_spectral_stats(theta_trans)
    
    print('\t-Scintillation depth', flush=True)
    stat_depth_scint = collect_spectral_stats(depth_scint)
    
    print('\t-Scintillation spot size', flush=True)
    stat_R_scint = collect_spectral_stats(R_scint[:,1])
    
    print('\t-Scintillation spot shift', flush=True)
    stat_dxy_scint = collect_spectral_stats(dxy_scint[:,1])
    
    print('\t-Number of scintillation photons', flush=True)
    stat_n_scint = collect_spectral_stats(n_scint)

    np.savez(directory + "/data/" + identifier + "/geant4_stats",
             n_eventType=n_eventType,
             stat_E_inc=stat_E_inc,
             stat_dE_trans=stat_dE_trans,
             stat_theta_trans=stat_theta_trans,
             stat_depth_scint=stat_depth_scint,
             stat_R_scint=stat_R_scint,
             stat_dxy_scint=stat_dxy_scint,
             stat_n_scint=stat_n_scint)

if __name__ == '__main__':
    identifier = 'YAGCe_150keV'
    
    E_inc, dE_trans, theta_trans, depth_scint, R_scint, dxy_scint, n_scint, n_rayl, n_phot, n_compt = merge_datafiles(identifier, 24)
    compute_stats(E_inc, dE_trans, theta_trans, depth_scint, R_scint, dxy_scint, n_scint, n_rayl, n_phot, n_compt)