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

def merge_datafiles(identifier, size, metric_type):
    if rank == 0:
        print('### Merging Data Files', end='', flush=True)
        for i in range(size):
            print(' | ' + str(i), end='', flush=True)
            while True:
                try:
                    with np.load(directory + "/data/" + identifier + "/geant4_data_" + str(i) + ".npz") as data:
                        if i == 0:
                            E_trans = data['E_inc'] - data['dE_trans']
                            n_scint = data['n_scint']
                            
                            n_rayl = data['n_rayl']
                            n_phot = data['n_phot']
                            n_compt = data['n_compt']
                        
                        else:
                            E_trans = np.hstack((data['E_inc'] - data['dE_trans'], E_trans))
                            n_scint = np.hstack((data['n_scint'], n_scint))
                            
                            n_rayl = np.hstack((data['n_rayl'], n_rayl))
                            n_phot = np.hstack((data['n_phot'], n_phot))
                            n_compt = np.hstack((data['n_compt'], n_compt))
                    break
                except:
                    time.sleep(1)
    
        print('', flush=True)
        
        if metric_type == 'n/E':
            metric = n_scint/E_trans
            metric[E_trans==0] = np.nan
        elif metric_type == 'n*E':
            metric = n_scint*E_trans
        elif metric_type == 'n_E':
            metric1 = n_scint.copy()
            metric2 = E_trans.copy()
    
        max_rayl = int(np.max(n_rayl)) + 1
        max_phot = int(np.max(n_phot)) + 1
        max_compt = int(np.max(n_compt)) + 1
        
        E_trans_range = np.array([np.nanmin(E_trans),np.nanmax(E_trans)])
        n_scint_range = np.array([np.nanmin(n_scint),np.nanmax(n_scint)])
        if metric_type == 'n/E' or metric_type == 'n*E':
            metric_range = np.array([np.nanmin(metric),np.nanmax(metric)])
        elif metric_type == 'n_E':
            metric1_range = np.array([np.nanmin(metric1),np.nanmax(metric1)])
            metric2_range = np.array([np.nanmin(metric2),np.nanmax(metric2)])
    
    else:
        max_rayl = 0
        max_phot = 0
        max_compt = 0
        
        E_trans_range = np.zeros(2, dtype=np.float64)
        n_scint_range = np.zeros(2, dtype=np.float64)
        if metric_type == 'n/E' or metric_type == 'n*E':
            metric_range = np.zeros(2, dtype=np.float64)
        elif metric_type == 'n_E':
            metric1_range = np.zeros(2, dtype=np.float64)
            metric2_range = np.zeros(2, dtype=np.float64)
    
    max_rayl = comm.bcast(max_rayl, root=0)
    max_phot = comm.bcast(max_phot, root=0)
    max_compt = comm.bcast(max_compt, root=0)
    
    comm.Bcast(E_trans_range, root=0)
    comm.Bcast(n_scint_range, root=0)
    if metric_type == 'n/E' or metric_type == 'n*E':
        comm.Bcast(metric_range, root=0)
    elif metric_type == 'n_E':
        comm.Bcast(metric1_range, root=0)
        comm.Bcast(metric2_range, root=0)
    
    if rank == 0:
        if metric_type == 'n/E' or metric_type == 'n*E':
            np.savez(directory + "/data/" + identifier + "/data_ranges_" + metric_type, max_rayl=max_rayl,
                                                                                        max_phot=max_phot,
                                                                                        max_compt=max_compt,
                                                                                        E_trans_range=E_trans_range,
                                                                                        n_scint_range=n_scint_range,
                                                                                        metric_range=metric_range)
        elif metric_type == 'n_E':
            np.savez(directory + "/data/" + identifier + "/data_ranges_" + metric_type, max_rayl=max_rayl,
                                                                                        max_phot=max_phot,
                                                                                        max_compt=max_compt,
                                                                                        E_trans_range=E_trans_range,
                                                                                        n_scint_range=n_scint_range,
                                                                                        metric1_range=metric1_range,
                                                                                        metric2_range=metric2_range)

    maxval = (max_rayl, max_phot, max_compt)
    if metric_type == 'n/E' or metric_type == 'n*E':
        ranges = (E_trans_range, n_scint_range, metric_range)
    elif metric_type == 'n_E':
        ranges = (E_trans_range, n_scint_range, metric1_range, metric2_range)

    return maxval, ranges

def compute_stats(identifier, size):

    maxval, ranges, counts, metrics = merge_datafiles(identifier, size)
    max_rayl, max_phot, max_compt = maxval
    E_trans_range, n_scint_range, I_scint_norm_range = ranges
    n_rayl, n_phot, n_compt = counts
    E_trans, n_scint, I_scint_norm = metrics

    if rank == 0:
        print('### Computing Relevant Statistics', flush=True)        
    
    E_step = np.ptp(E_inc_range)/(n_E_inc - 1)

    mask_rayl = np.zeros((n_rayl.size, max_rayl))
    for i in range(max_rayl):
        mask_rayl[:,i] = np.abs(n_rayl - i) < 0.1
    
    mask_phot = np.zeros((n_phot.size, max_phot))
    for i in range(max_phot):
        mask_phot[:,i] = np.abs(n_phot - i) < 0.1
    
    mask_compt = np.zeros((n_compt.size, max_compt))
    for i in range(max_compt):
        mask_compt[:,i] = np.abs(n_compt - i) < 0.1
    
    if rank == 0:
        print('\t-Number of occurence', flush=True)
    n_eventType = np.zeros((max_rayl, max_phot, max_compt, n_E_inc))
    for i in range(max_rayl):
        for j in range(max_phot):
            for k in range(max_compt):
                for l in range(n_E_inc):
                    n_eventType[i,j,k,l] = np.sum(mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]*(np.abs(E_inc - (E_step*l + np.min(E_inc))) < E_step/2))
    
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
        stat = np.zeros((max_rayl, max_phot, max_compt, n_E_inc, 2))
        nanmask = (1 - np.isnan(raw)).astype(bool)
        for i in range(max_rayl):
            for j in range(max_phot):
                for k in range(max_compt):
                    for l in range(n_E_inc):
                        mask = (mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]*(np.abs(E_inc - (E_step*l + np.min(E_inc))) < E_step/2)).astype(bool)
                        
                        raw_masked = raw[mask*nanmask]
                        if raw_masked.size > 0:
                            stat[i,j,k,l,0] = np.mean(raw[mask*nanmask])
                            stat[i,j,k,l,1] = np.std(raw[mask*nanmask])
                        else:
                            stat[i,j,k,l,0] = np.nan
                            stat[i,j,k,l,1] = np.nan
        
        return stat
    
    def collect_percentiles(raw):
        stat = np.zeros((max_rayl, max_phot, max_compt, 5))
        nanmask = (1 - np.isnan(raw)).astype(bool)
        for i in range(max_rayl):
            for j in range(max_phot):
                for k in range(max_compt):
                    mask = (mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]).astype(bool)
                    
                    raw_masked = raw[mask*nanmask]
                    if raw_masked.size > 0:
                        stat[i,j,k,0] = np.min(raw[mask*nanmask])
                        stat[i,j,k,1] = np.percentile(raw[mask*nanmask], 25)
                        stat[i,j,k,2] = np.percentile(raw[mask*nanmask], 50)
                        stat[i,j,k,3] = np.percentile(raw[mask*nanmask], 75)
                        stat[i,j,k,4] = np.max(raw[mask*nanmask])
                    else:
                        stat[i,j,k,:] = np.nan
        
        return stat
    
    if rank == 0:
        print('\t-Energy loss', flush=True)
    stat_E_trans = collect_percentiles(E_inc - dE_trans)
    
    if rank == 0:
        print('\t-Number of scintillation photons', flush=True)
    stat_n_scint = collect_percentiles(n_scint)
    
    if rank == 0:
        print('\t-Energy-normalized scintillation intensity', flush=True)
    stat_I_scint_norm = collect_percentiles(n_scint/(E_inc - dE_trans))

    np.savez(directory + "/data/" + identifier + "/geant4_stats_" + str(rank),
             n_data=E_inc.size,
             stat_I_scint_norm=stat_I_scint_norm,
             stat_E_trans=stat_E_trans,
             stat_n_scint=stat_n_scint,
             n_eventType=n_eventType)
    
    comm.Barrier()

def compute_dist(identifier, size, subgroup_size, metric_type):

    maxval, ranges = merge_datafiles(identifier, size, metric_type)
    max_rayl, max_phot, max_compt = maxval
    if metric_type == 'n/E' or metric_type == 'n*E':
        E_trans_range, n_scint_range, metric_range = ranges
    elif metric_type == 'n_E':
        E_trans_range, n_scint_range, metric1_range, metric2_range = ranges

    if rank == 0:
        print('### Computing Event Distribution', flush=True)
    
    for i in range(subgroup_size):
        with np.load(directory + "/data/" + identifier + "/geant4_data_" + str(subgroup_size*rank + i) + ".npz") as data:
            if i == 0:
                E_trans = data['E_inc'] - data['dE_trans']
                n_scint = data['n_scint']
                theta_trans = data['theta_trans']
                
                n_rayl = data['n_rayl']
                n_phot = data['n_phot']
                n_compt = data['n_compt']
            
            else:
                E_trans = np.hstack((data['E_inc'] - data['dE_trans'], E_trans))
                n_scint = np.hstack((data['n_scint'], n_scint))
                theta_trans = np.hstack((data['theta_trans'], theta_trans))
                
                n_rayl = np.hstack((data['n_rayl'], n_rayl))
                n_phot = np.hstack((data['n_phot'], n_phot))
                n_compt = np.hstack((data['n_compt'], n_compt))
    
    if metric_type == 'n/E':
        metric = n_scint/E_trans
        metric[E_trans==0] = np.nan
    elif metric_type == 'n*E':
        metric = n_scint*E_trans
    elif metric_type == 'n_E':
        metric1 = n_scint.copy()
        metric2 = E_trans.copy()
        metric3 = theta_trans.copy()

    mask_rayl = np.zeros((n_rayl.size, max_rayl))
    for i in range(max_rayl):
        mask_rayl[:,i] = np.abs(n_rayl - i) < 0.1
    
    mask_phot = np.zeros((n_phot.size, max_phot))
    for i in range(max_phot):
        mask_phot[:,i] = np.abs(n_phot - i) < 0.1
    
    mask_compt = np.zeros((n_compt.size, max_compt))
    for i in range(max_compt):
        mask_compt[:,i] = np.abs(n_compt - i) < 0.1

    def collect_distribution_1param(raw, raw_range, ndisc):
        step = np.ptp(raw_range)/ndisc
        dist = np.zeros((max_rayl, max_phot, max_compt, ndisc))
        
        cnt = 0
        for i in range(max_rayl):
            for j in range(max_phot):
                for k in range(max_compt):
                    for l in range(ndisc):
                        t1 = time.time()
                        mask = (mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]\
                                *(raw >= l*step)*(raw < (l+1)*step)).astype(bool)
                                
                        dist[i,j,k,l] = np.sum(mask)
                        t2 = time.time()
                        cnt += 1
                        if rank == 0:
                            print('Time Remaining: ' + str(np.round((max_rayl*max_phot*max_compt*ndisc - cnt)/3600, 2)) + ' hrs', flush=True)
        
        raw_list = np.linspace(raw_range[0], raw_range[1], ndisc+1)
        
        return dist, raw_list
    
    def collect_distribution_2param(raw1, raw2, raw1_range, raw2_range, ndisc1, ndisc2):
        step1 = np.ptp(raw1_range)/ndisc1
        step2 = np.ptp(raw2_range)/ndisc2
        dist = np.zeros((max_rayl, max_phot, max_compt, ndisc1, ndisc2))
        
        cnt = 0
        for i in range(max_rayl):
            for j in range(max_phot):
                for k in range(max_compt):
                    for l in range(ndisc1):
                        for m in range(ndisc2):
                            t1 = time.time()
                            mask = (mask_rayl[:,i]*mask_phot[:,j]*mask_compt[:,k]\
                                    *(raw1 >= l*step1)*(raw1 < (l+1)*step1)\
                                    *(raw2 >= l*step2)*(raw2 < (l+1)*step2)).astype(bool)
                                    
                            dist[i,j,k,l,m] = np.sum(mask)
                            t2 = time.time()
                            cnt += 1
                            if rank == 0:
                                print('Time Remaining: ' + str(np.round((max_rayl*max_phot*max_compt*ndisc1*ndisc2 - cnt)/3600, 2)) + ' hrs', flush=True)
        
        raw1_list = np.linspace(raw1_range[0], raw1_range[1], ndisc1+1)
        raw2_list = np.linspace(raw2_range[0], raw2_range[1], ndisc2+1)
        
        return dist, raw1_list, raw2_list
    
    if metric_type == 'n/E' or metric_type == 'n*E':
        ndisc = 101
        dist, raw_list = collect_distribution_1param(metric, metric_range, ndisc)

        np.savez(directory + "/data/" + identifier + "/geant4_dist_" + metric_type + "_" + str(rank),
                 dist=dist,
                 raw_list=raw_list)
                 
    elif metric_type == 'n_E':
        ndisc1 = 31
        ndisc2 = 31
        dist, raw1_list, raw2_list = collect_distribution_2param(metric1, metric2, metric1_range, metric2_range, ndisc1, ndisc2)

        np.savez(directory + "/data/" + identifier + "/geant4_dist_" + metric_type + "_" + str(rank),
                 dist=dist,
                 raw1_list=raw1_list,
                 raw2_list=raw2_list,
                 )
        mask_compt = (n_phot == 0)*(n_compt > 0)
        np.savez(directory + "/data/" + identifier + "/geant4_corr_" + metric_type + "_" + str(rank),
                 n_compt=n_compt[mask_compt],
                 metric1=metric1[mask_compt],
                 metric2=metric2[mask_compt],
                 metric3=metric3[mask_compt],
                 )
             
    comm.Barrier()             

    if rank == 0:
        print('### Merging Distributions', end='', flush=True)
        
        if metric_type == 'n/E' or metric_type == 'n*E':
            dist = np.zeros((max_rayl, max_phot, max_compt, ndisc))
        else:
            dist = np.zeros((max_rayl, max_phot, max_compt, ndisc1, ndisc2))
        
        for i in range(int(size/subgroup_size)):
            print(' | ' + str(i), end='', flush=True)
            with np.load(directory + "/data/" + identifier + "/geant4_dist_" + metric_type + "_" + str(i) + ".npz") as data:
                dist += data['dist']
        
        if metric_type == 'n/E' or metric_type == 'n*E':
            np.savez(directory + "/data/" + identifier + "/geant4_dist_merged_" + metric_type,
                     dist=dist,
                     raw_list=raw_list)
        
        elif metric_type == 'n_E':
            np.savez(directory + "/data/" + identifier + "/geant4_dist_merged_" + metric_type,
                     dist=dist,
                     raw1_list=raw1_list,
                     raw2_list=raw2_list)
    
        print('', flush=True)

if __name__ == '__main__':
    identifier = 'YAGCe_50_120keV_tube_Pb3mm'
    
    #compute_stats(max_rayl, max_phot, max_compt, E_inc_range, 14, n_scint, E_inc, dE_trans, n_rayl, n_phot, n_compt)
    compute_dist(identifier, 100, 10, 'n_E')
    print('Done', flush=True)