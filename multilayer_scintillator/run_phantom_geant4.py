import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numba import jit
import autograd.numpy as npa
from scipy.optimize import least_squares, Bounds, minimize, linear_sum_assignment
from scipy.cluster.vq import kmeans2
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, anderson, gaussian_kde
from scipy.io import loadmat
from scipy.interpolate import interpn
from PIL import Image, ImageChops
import diptest
import matplotlib.pyplot as plt
from itertools import product
import shutil
import subprocess
import uproot
import time
import util.read_txt as txt
import multilayer_scintillator.xray_spectrum_generator as xray
import multilayer_scintillator.generate_linAttCoeff_dat as phantom
import multilayer_scintillator.filtered_backprojection as FBP

from mpi4py import MPI
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

class geant4:
    def __init__(self,
                 G4directory,
                 RGB_scintillator,
                 sampleID,
                 dimSampleVoxel,
                 indSampleVoxel,
                 IConc,
                 GdConc,
                 dimScint,
                 dimDetVoxel,
                 NDetVoxel,
                 gapSampleScint,
                 gapScintDet,
                 Nlayers,
                 scintillator_material,
                 scintillator_thickness,
                 check_overlap,
                 energy_range,
                 energy_tgt,
                 energy_tgt_index_imed,
                 energy_bins,
                 d_mean_bins,
                 theta_filt,
                 xray_target_material=None,
                 xray_operating_V=None,
                 xray_tilt_theta=None,
                 xray_filter_material=None,
                 xray_filter_thickness=None,
                 verbose=False,
                 distribution_datafile = '',
                 identifier='',
                 rank_start=0,
                 ):
                 
        self.G4directory = G4directory
        self.RGB_scintillator = RGB_scintillator
        
        # Sample
        self.sampleID = sampleID
        self.dimSampleVoxel = dimSampleVoxel
        self.indSampleVoxel = indSampleVoxel
        self.NSampleVoxel = indSampleVoxel[:,1] - indSampleVoxel[:,0]
        self.dimSample = self.NSampleVoxel*dimSampleVoxel # cm
        self.IConc = IConc
        self.GdConc = GdConc
        
        # Scintillator
        self.dimScint = dimScint # cm
        self.Nlayers = Nlayers
        self.scintillator_material = scintillator_material
        self.scintillator_thickness = scintillator_thickness
        
        # Detector
        self.dimDetVoxel = dimDetVoxel
        self.NDetVoxel = NDetVoxel
        self.dimDet = NDetVoxel*dimDetVoxel # um
        self.image_reconstr = np.zeros((NDetVoxel[0], NDetVoxel[1], energy_tgt.size-1))
        self.image_true = np.zeros((NDetVoxel[0], NDetVoxel[1], energy_tgt.size-1))
        
        # World
        self.dimWorld = np.max(np.vstack((dimScint, self.dimSample[:2], self.dimDet[:2]/1e4)), axis=0) # cm
        self.gapSampleScint = gapSampleScint
        self.gapScintDet = gapScintDet
        self.dzWorld = 1.01*(0.1 + self.dimSample[2] + gapSampleScint + np.sum(self.scintillator_thickness)/10 + gapScintDet + self.dimDet[2]/1e4) # cm
        
        self.check_overlap = check_overlap
        
        self.energy_list = np.linspace(energy_range[0], energy_range[1], energy_bins)
        self.energy_bins = energy_bins
        self.energy_tgt = energy_tgt
        self.energy_tgt_index_imed = energy_tgt_index_imed
        self.energy_tgt_ind = np.zeros(energy_tgt.size).astype(int)
        for i_tgt in range(energy_tgt.size):
            self.energy_tgt_ind[i_tgt] = np.argmin(np.abs(self.energy_list - self.energy_tgt[i_tgt]))
        
        self.sensRGB = txt.read_txt(directory + "/data/rgb_sensitivity")
        self.theta_filt = theta_filt
        if theta_filt is None:
            self.T_filt = txt.read_txt(directory + "/data/angle_filter_Gadox_NaI_T")
            self.wvl_list_filt = np.squeeze(txt.read_txt(directory + "/data/angle_filter_wvl"))
            self.theta_list_filt = np.squeeze(txt.read_txt(directory + "/data/angle_filter_theta")*np.pi/180)
        self.nNaI = txt.read_txt(directory + "/data/rindexNaI")
        self.nGadox = txt.read_txt(directory + "/data/rindexGadox")
        
        self.energy_list, xraySrc = xray.get_spectrum(xray_target_material,
                                                      self.energy_list,
                                                      xray_operating_V,
                                                      xray_tilt_theta,
                                                      xray_filter_material,
                                                      xray_filter_thickness,
                                                      )
        self.energy_list = np.delete(self.energy_list, np.where(xraySrc==0)[0])
        xraySrc = np.delete(xraySrc, np.where(xraySrc==0)[0])
        
        self.energy_bins = self.energy_list.size
        self.xraySrc = xraySrc/np.sum(xraySrc)
        
        fig, ax = plt.subplots(1, 1, figsize=[16,12], dpi=300)
        ax.plot(self.energy_list, self.xraySrc, linewidth=2)
        plt.savefig(directory + "/plots/" + identifier + "_xraySrc.png", dpi=300)
        plt.close()
        
        self.verbose = verbose
        self.distribution_datafile = distribution_datafile
        self.identifier = identifier
        self.rank_start = rank_start
        
        if self.RGB_scintillator:
            self.d_mean_list = np.linspace(0, np.max(self.dimDet[:2])/5e4, d_mean_bins) # mm
        else:
            self.d_mean_list = np.linspace(0, np.max(self.dimDet[:2])/5e4, d_mean_bins) # mm
        self.d_mean_bins = d_mean_bins
        self.n_ph_bins = 201
        self.n_ph_list = np.linspace(0, 200, self.n_ph_bins)
        # d_mean Limits
        # RGB150v3: 1e-2~5e3
        # 0.3mm or filter or Gadox25deg or GadoxFilter: 1e-2~5e4
        #    NaI30deg or Gadox20deg: 7.5e4
        #    NaI25deg or Gadox15deg: 1e5
        #    NaI20deg: 1.25e5
        #    NaI15deg: 1.5e5
        #    NaI10deg: 2.5e5
        #    NaI5deg: 5e5
        # 0.6mm: 2.5e4
        # 1.5mm: 1e4
        # 3.0mm: 5e3
        
        if os.path.exists(directory + "/data/energy_distribution_data_" + distribution_datafile + ".npz"):
            self.energy_distribution = np.load(directory + "/data/energy_distribution_data_" + distribution_datafile + ".npz")['energy_distribution']
            try:
                self.n_photon_distribution = np.load(directory + "/data/energy_distribution_data_" + distribution_datafile + ".npz")['n_photon_distribution']
            except:
                pass
            if rank == 0:
                print('Energy distribution loaded', flush=True)
        else:
            self.energy_distribution = None
            self.n_photon_distribution = None
            if rank == 0:
                print('No energy distribution found', flush=True)
    
    def cmake(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["cmake", ".."])
        time.sleep(20)
        
    def make(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["make", "-j4"])
        time.sleep(40)
    
    def write_dat_file(self, name, data):
        with open(self.G4directory + "/build/" + name + ".dat", "w") as dat:
            for nd in range(data.size-1):
                dat.write(data[nd] + "\n")
            dat.write(data[-1])
    
    def make_simulation_macro(self, Nruns, source_energy=None, source_xy=None):
        with open(self.G4directory + "/build/run_detector_rank" + str(comm.rank + self.rank_start) + ".mac", "w") as mac:
            mac.write("/tracking/verbose 2\n")
        
            mac.write("/system/root_file_name output" + str(comm.rank + self.rank_start) + ".root\n")
        
            mac.write("/structure/xWorld " + str(np.round(self.dimWorld[0], 6)) + "\n")
            mac.write("/structure/yWorld " + str(np.round(self.dimWorld[1], 6)) + "\n")
            mac.write("/structure/zWorld " + str(np.round(self.dzWorld, 6)) + "\n\n")
            mac.write("/structure/gapSampleScint " + str(np.round(self.gapSampleScint, 6)) + "\n")
            mac.write("/structure/gapScintDet " + str(np.round(self.gapScintDet, 6)) + "\n")
            
            mac.write("/structure/xScint " + str(np.round(self.dimScint[0], 6)) + "\n")
            mac.write("/structure/yScint " + str(np.round(self.dimScint[1], 6)) + "\n")
            
            mac.write("/structure/nLayers " + str(int(self.Nlayers)) + "\n")
            
            for nl in range(self.Nlayers):
                mac.write("/structure/scintillatorThickness" + str(nl+1) + " %.6f\n" %(self.scintillator_thickness[nl]))
            
            for nl in range(self.Nlayers):
                mac.write("/structure/scintillatorMaterial" + str(nl+1) + " " + str(int(self.scintillator_material[nl])) + "\n")
            
            if self.theta_filt is None:
                mac.write("/structure/angleFilter true\n")
                mac.write("/structure/angleFilterStart 101\n")
            else:
                mac.write("/structure/angleFilter false\n")
            mac.write("\n")
            
            mac.write("/structure/sampleID " + str(int(self.sampleID)) + "\n")
            mac.write("/structure/xSample " + str(np.round(self.dimSample[0], 6)) + "\n")
            mac.write("/structure/ySample " + str(np.round(self.dimSample[1], 6)) + "\n")
            mac.write("/structure/zSample " + str(np.round(self.dimSample[2], 6)) + "\n")
            if self.sampleID == 1:
                mac.write("/structure/nSampleXall 254\n")
            elif self.sampleID == 2:
                mac.write("/structure/nSampleXall 299\n")
            mac.write("/structure/nSampleX " + str(int(self.NSampleVoxel[0])) + "\n")
            mac.write("/structure/nSampleY " + str(int(self.NSampleVoxel[1])) + "\n")
            mac.write("/structure/nSampleZ " + str(int(self.NSampleVoxel[2])) + "\n")
            mac.write("/structure/indSampleX " + str(int(self.indSampleVoxel[0,0])) + "\n")
            mac.write("/structure/indSampleY " + str(int(self.indSampleVoxel[1,0])) + "\n")
            mac.write("/structure/indSampleZ " + str(int(self.indSampleVoxel[2,0])) + "\n")
            mac.write("/structure/IConcentration " + str(self.IConc) + "\n")
            mac.write("/structure/GdConcentration " + str(self.GdConc) + "\n")
            mac.write("\n")
            
            mac.write("/structure/constructDetectors 1\n")
            mac.write("/structure/xDet " + str(np.round(self.dimDet[0], 6)) + "\n")
            mac.write("/structure/yDet " + str(np.round(self.dimDet[1], 6)) + "\n")
            mac.write("/structure/nDetX " + str(int(self.NDetVoxel[0])) + "\n")
            mac.write("/structure/nDetY " + str(int(self.NDetVoxel[1])) + "\n")
            mac.write("/structure/detectorDepth " + str(np.round(self.dimDet[2], 6)) + "\n\n")
            
            mac.write("/structure/checkDetectorsOverlaps " + str(int(self.check_overlap)) + "\n\n")
            
            mac.write("/run/numberOfThreads 1\n")
            mac.write("/run/initialize\n")
            
            mac.write("/gps/particle gamma\n")
            mac.write("/gps/direction 0 0 1\n")
            if source_xy is None:
                mac.write("/gps/pos/type Plane\n")
                mac.write("/gps/pos/shape Square\n")
                mac.write("/gps/pos/centre 0 0 " + str(np.round(-self.dzWorld/2.00, 6)) + " cm\n")
                mac.write("/gps/pos/halfx " + str(np.round(self.dimDet[0]/2e4, 6)) + " cm\n")
                mac.write("/gps/pos/halfy " + str(np.round(self.dimDet[1]/2e4, 6)) + " cm\n")
            elif source_xy == 'center':
                mac.write("/gps/pos/type Point\n")
                mac.write("/gps/pos/centre 0 0 " + str(np.round(-self.dzWorld/2.00, 6)) + " cm\n")
                
            if source_energy is None:
                mac.write("/gps/ene/type User\n")
                mac.write("/gps/ene/emspec 0\n")
                mac.write("/gps/hist/type energy\n")
                mac.write("/gps/hist/point 0. 0.\n")
                for ne in range(self.energy_bins):
                    mac.write("/gps/hist/point " + str(np.round(self.energy_list[ne]/1e3, 6)) + " " + str(np.round(self.xraySrc[ne], 6)) + "\n")
            else:
                mac.write("/gps/ene/type Mono\n")
                mac.write("/gps/ene/mono " + str(np.round(source_energy, 6)) + " keV\n")
            mac.write("/run/beamOn " + str(int(Nruns)))
        
    def read_output(self, Nruns):
        if os.path.exists(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + ".root"):
            os.remove(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + ".root")
    
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["./NS", "run_detector_rank" + str(comm.rank + self.rank_start) + ".mac"])
        
        while True:
            if os.path.exists(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + ".root"):
                break
            time.sleep(1)
        
        while True:
            try:
                root_file = uproot.open(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + ".root")
                photons = root_file['Photons']
                hits = root_file['Hits']
                source = root_file['Source']
                debug = root_file['Debug']
                debug2 = root_file['Debug2']
                break
            except:
                time.sleep(1)
        
        event_debug = np.array(debug["fEvent"].array())
        vz_debug = np.array(debug["vZ"].array())
        wvl_debug = np.array(debug["wlen"].array())
        proc_debug = np.array(debug["process"].array())
        
        event_debug2 = np.array(debug2["fEvent"].array())
        mz_debug2 = np.array(debug2["mZ"].array())
        wvl_debug2 = np.array(debug2["wlen"].array())
        preVol_debug2 = np.array(debug2["preVol"].array())
        postVol_debug2 = np.array(debug2["postVol"].array())
        
        creatorProcess = [t for t in photons["fCreatorProcess"].array()] # run 10000 or more at a time and use fEvent to determine which is which
        scint_photon_ind = np.array([t == 'Scintillation' for t in creatorProcess])
        
        np.savez(directory + "/debug_vZ", event_debug=event_debug, vz_debug=vz_debug, wvl_debug=wvl_debug, proc_debug=proc_debug,
                 event_debug2=event_debug2, mz_debug2=mz_debug2, wvl_debug2=wvl_debug2, preVol_debug2=preVol_debug2, postVol_debug2=postVol_debug2,
                 creatorProcess=np.array(creatorProcess), eventHits=np.array(hits["fEvent"].array()))
        assert False
        
        wvl_list = np.array(photons["fWlen"].array())[scint_photon_ind]
        px_list = np.array(photons["pX"].array())[scint_photon_ind]
        py_list = np.array(photons["pY"].array())[scint_photon_ind]
        pz_list = np.array(photons["pZ"].array())[scint_photon_ind]
        vz_list = np.array(photons["vZ"].array())[scint_photon_ind]
        pxy_list = np.sqrt(px_list**2 + py_list**2)
        theta_air = np.arctan2(pxy_list, pz_list)
        nScint_list = np.interp(wvl_list, self.nGadox[:,0], self.nGadox[:,1])
        theta_Scint = np.arcsin(np.sin(theta_air)/nScint_list)

        mask = vz_list > (-self.dzWorld/2 + self.dimSample[2] + self.gapSampleScint)*1e1 + np.sum(self.scintillator_thickness[:2]) # mm
        if self.theta_filt is not None:
            mask += theta_Scint < self.theta_filt
        else:
            T_interp = interpn((self.wvl_list_filt, self.theta_list_filt), self.T_filt, np.vstack((wvl_list, theta_Scint)).T)
            mask += np.random.rand(wvl_list.size) < T_interp
        mask *= (wvl_list >= 303)*(wvl_list <= 1100)
        wvl_list = wvl_list[mask]
        
        eventHits = np.array(hits["fEvent"].array())[scint_photon_ind][mask]
        x_list = np.array(hits["fX"].array())[scint_photon_ind][mask]
        y_list = np.array(hits["fY"].array())[scint_photon_ind][mask]
        xy_list = np.vstack((x_list, y_list)).T + self.dimDet[np.newaxis,:2]*1e-3/2 # mm
        
        energySrc_list = np.array(source["fEnergy"].array())*1e3 # keV
        xSrc_list = np.array(source["fX"].array())
        ySrc_list = np.array(source["fY"].array())
        xySrc_list = np.vstack((xSrc_list, ySrc_list)).T + self.dimDet[np.newaxis,:2]*1e-3/2 # mm
        
        wvl_dict = {}
        xy_dict = {}
        for nr in range(Nruns):
            wvl_dict[nr] = wvl_list[eventHits==nr]
            xy_dict[nr] = xy_list[eventHits==nr,:]
        
        #if comm.rank == 0:
        np.savez(directory + "/data/debug_read_output" + str(comm.rank + self.rank_start), wvl_list=wvl_list, eventHits=eventHits, xy_list=xy_list, energySrc_list=energySrc_list,
                 xySrc_list=xySrc_list, px_list=px_list, py_list=py_list, pz_list=pz_list, theta_air=theta_air, mask=mask, nScint_list=nScint_list, theta_Scint=theta_Scint)
        
        return wvl_dict, xy_dict, energySrc_list, xySrc_list
    
    def RGB_readings(self, wvl_list):
        RGB = np.zeros((wvl_list.size, 3))
        prob = np.zeros((wvl_list.size, 3))
        for i in range(3):
            prob[:,i] = np.interp(wvl_list, self.sensRGB[:,0], self.sensRGB[:,i+1])
        for i in range(wvl_list.size):
            try:
                index = np.random.choice(np.arange(3), p=prob[i,:]/np.sum(prob[i,:]))
            except:
                print(wvl_list[i])
                print(prob[i,:])
            if index < 3:
                RGB[i,index] = 1
        
        return RGB
    
    def data_interpretation(self, xy_list):
        xy_local_density = np.zeros(xy_list.shape[0])
        for i in range(xy_list.shape[0]):
            dist = np.sqrt(np.sum((xy_list[i,:][np.newaxis,:] - xy_list)**2, axis=1))
            xy_local_density[i] = np.sum(dist < self.dimDet[0]/5e4)
        
        xy_list = xy_list[xy_local_density>np.max(xy_local_density)/100,:]
    
        centroid, label = weighted_kmeans(xy_list, np.ones(xy_list.shape[0]), nCluster=1, maxiter=20, centroid=(self.dimDet[:2]/1e3).reshape(1,-1)) # weights-->RGB_list
    
        xy_temp = np.array(list(product(np.linspace(np.min(xy_list[:,0]), np.max(xy_list[:,0]), int(1e2)),
                                        np.linspace(np.min(xy_list[:,1]), np.max(xy_list[:,1]), int(1e2)))))
        
        try:
            kernel = gaussian_kde(xy_list.T)
            kde = kernel(xy_list.T)
            kde_temp = kernel(xy_temp.T)
            
            npts = np.sum(kde>np.max(np.hstack((kde, kde_temp)))/10)
            d_mean = np.sum(np.sqrt(np.sum((centroid[0,:][np.newaxis,:] - xy_list[kde>np.max(np.hstack((kde, kde_temp)))/10,:])**2, axis=1)))/npts
        except:
            np.savez(directory + "/data/debug_kde", xy_list=xy_list)
            
            npts = xy_list.shape[0]
            d_mean = np.sum(np.sqrt(np.sum((centroid[0,:][np.newaxis,:] - xy_list)**2, axis=1)))/npts
        
        ind_d_mean = np.argmin(np.abs(d_mean - self.d_mean_list))
        
#        if comm.rank == 0:
#            np.savez(directory + "/data/debug_data_interp_" + str(int(xy_list.size/2)), xy_list=xy_list,
#                     centroid=centroid, label=label, d_mean=d_mean, ind_d_mean=ind_d_mean, d_mean_list=self.d_mean_list)
    
        return ind_d_mean, d_mean
    
    def reconstruct_image(self, energy_distribution_norm, source_energy_list, source_xy_list, xy_list, RGB_list, event_label, Nphoton_rate):
        xy_local_density = np.zeros(xy_list.shape[0])
        for i in range(xy_list.shape[0]):
            dist = np.sqrt(np.sum((xy_list[i,:][np.newaxis,:] - xy_list)**2, axis=1))
            xy_local_density[i] = np.sum(dist < self.dimDet[0]/5e4)
        
        xy_list = xy_list[xy_local_density>np.max(xy_local_density)/100,:]
        if self.RGB_scintillator:
            RGB_list = RGB_list[xy_local_density>np.max(xy_local_density)/100,:]
        event_label = event_label[xy_local_density>np.max(xy_local_density)/100]
    
        min_cluster_size = 3 # 100,200,200
        max_cluster_size = 2000 # 300,600,600
    
        if xy_list.shape[0] > min_cluster_size:
            centroid, label = self.gmeans(xy_list, 20, 20, min_cluster_size, max_cluster_size)
            nk = centroid.shape[0]
            fm = fowlkes_mallows_index(event_label, label)
            
            d_mean = np.zeros(nk)
            xrayInc_all = np.zeros((self.energy_tgt.size-1, nk))
            if self.RGB_scintillator:
                ind_RGB = np.zeros(nk)
        
            for k in range(nk):
                npts = int(np.sum(label==k))
                if npts > min_cluster_size:
                    xy_temp = xy_list[label==k,:]
                    xy_temp2 = np.array(list(product(np.linspace(np.min(xy_temp[:,0]), np.max(xy_temp[:,0]), int(1e2)),
                                                     np.linspace(np.min(xy_temp[:,1]), np.max(xy_temp[:,1]), int(1e2)))))
                
                    try:
                        kernel = gaussian_kde(xy_temp.T)
                    except:
                        np.savez(directory + "/data/debug_kde", xy_temp=xy_temp)
                        continue
                
                    kde = kernel(xy_temp.T)
                    kde_temp = kernel(xy_temp2.T)
                    npts = np.sum(kde>np.max(np.hstack((kde, kde_temp)))/10)
            
                    d_mean[k] = np.sum(np.sqrt(np.sum((np.tile(centroid[k,:], (npts,1)) - xy_temp[kde>np.max(np.hstack((kde, kde_temp)))/10,:])**2, axis=1)))/npts
                    ind_d_mean = np.argmin(np.abs(d_mean[k] - self.d_mean_list))
                    if self.RGB_scintillator:
                        RGB_temp = RGB_list[label==k,:]
                        ind_RGB[k] = np.argmax(np.mean(RGB_temp, axis=0))
                        xrayInc_all[:,k] = energy_distribution_norm[int(ind_RGB[k]),ind_d_mean,:]
                    else:
                        xrayInc_all[:,k] = energy_distribution_norm[ind_d_mean,:]
                
                    ind_x0 = int((centroid[k,0] - self.dimDetVoxel[0]/2e3)//(self.dimDetVoxel[0]/1e3))
                    x0 = self.dimDetVoxel[0]/2e3 + ind_x0*self.dimDetVoxel[0]/1e3
                    ind_x1 = ind_x0 + 1
                    x1 = self.dimDetVoxel[0]/2e3 + ind_x1*self.dimDetVoxel[0]/1e3
                    if ind_x1 == self.NDetVoxel[0]:
                        ind_x0 -= 1
                        ind_x1 -= 1
                        
                    ind_y0 = int((centroid[k,1] - self.dimDetVoxel[1]/2e3)//(self.dimDetVoxel[1]/1e3))
                    y0 = self.dimDetVoxel[1]/2e3 + ind_y0*self.dimDetVoxel[1]/1e3
                    ind_y1 = ind_y0 + 1
                    y1 = self.dimDetVoxel[1]/2e3 + ind_y1*self.dimDetVoxel[1]/1e3
                    if ind_y1 == self.NDetVoxel[1]:
                        ind_y0 -= 1
                        ind_y1 -= 1
        
                    ratio = np.zeros(4)
                    ratio[0] = np.sqrt(np.sum((np.array([x0,y0]) - centroid[k,:])**2))
                    ratio[1] = np.sqrt(np.sum((np.array([x0,y1]) - centroid[k,:])**2))
                    ratio[2] = np.sqrt(np.sum((np.array([x1,y0]) - centroid[k,:])**2))
                    ratio[3] = np.sqrt(np.sum((np.array([x1,y1]) - centroid[k,:])**2))
                    if np.any(ratio == 0):
                        minArg = np.argmin(ratio)
                        ratio = np.zeros(4)
                        ratio[minArg] = 1
                    else:
                        ratio = 1/ratio
                        ratio /= np.sum(ratio)
        
                    self.image_reconstr[ind_x0,ind_y0,:] += xrayInc_all[:,k]*ratio[0]
                    self.image_reconstr[ind_x0,ind_y1,:] += xrayInc_all[:,k]*ratio[1]
                    self.image_reconstr[ind_x1,ind_y0,:] += xrayInc_all[:,k]*ratio[2]
                    self.image_reconstr[ind_x1,ind_y1,:] += xrayInc_all[:,k]*ratio[3]
            
            if comm.rank == 0:
                cnt = 0
                while True:
                    if not os.path.exists(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(Nphoton_rate) + "_" + str(cnt) + ".npz"):
                        if self.RGB_scintillator:
                            np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(Nphoton_rate) + "_" + str(cnt), d_mean=d_mean,
                                 xrayInc=xrayInc_all, ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, xy_list=xy_list,
                                 RGB_list=RGB_list, source_energy_list=source_energy_list, source_xy_list=source_xy_list, centroid=centroid, label=label, event_label=event_label,
                                 ind_RGB=ind_RGB, fm=fm, energy_distribution_norm=energy_distribution_norm)
                        else:
                            np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(Nphoton_rate) + "_" + str(cnt), d_mean=d_mean,
                                     xrayInc=xrayInc_all, ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, centroid=centroid,
                                     xy_list=xy_list, label=label, source_energy_list=source_energy_list, source_xy_list=source_xy_list, event_label=event_label, fm=fm,
                                     energy_distribution_norm=energy_distribution_norm)
                        break
                    if cnt >= 10:
                        break
                    cnt += 1
        else:
            fm = np.nan
    
        return fm
    
    def gmeans(self, xy_list, maxiter_gmeans, maxiter_kmeans, min_cluster_size=1000, max_cluster_size=1000, p_crit=0.05, savefilename=''):
        centroid = np.mean(xy_list, axis=0).reshape(1, -1)
        label = np.zeros(xy_list.shape[0])
        
        cnt = 0
        while True:
            centroid_prev = centroid.copy()
            label_prev = label.copy()
            mask_remove = np.zeros(centroid_prev.shape[0]).astype(bool)
            centroid_temp = None
            for nC in range(centroid_prev.shape[0]):
                xy_nC = xy_list[label_prev==nC,:]
            
                if xy_nC.shape[0] > min_cluster_size:
                    eigval, eigvec = pca(xy_nC)
                    centroid_temp1 = centroid_prev[nC,:] - eigvec*np.sqrt(2*eigval/np.pi)
                    centroid_temp2 = centroid_prev[nC,:] + eigvec*np.sqrt(2*eigval/np.pi)
                    
                    if centroid_temp is None:
                        centroid_temp = np.vstack((centroid_temp1, centroid_temp2))
                        eigvec_all = eigvec.reshape(1, -1)
                    else:
                        centroid_temp = np.vstack((centroid_temp, centroid_temp1, centroid_temp2))
                        eigvec_all = np.vstack((eigvec_all, eigvec))
                else:
                    mask_remove[nC] = True
            centroid_prev = centroid_prev[~mask_remove,:]
            
            centroid_temp, label_temp = weighted_kmeans(xy_list, np.ones(xy_list.shape[0]), centroid_temp.shape[0], 0, centroid=centroid_temp)
            
            ind_label = 0
            split_check = False
            centroid = None
            pval_save = np.zeros(centroid_prev.shape[0])
            for nC in range(centroid_prev.shape[0]):
                mask_nC = np.isin(label_temp, np.array([2*nC,2*nC+1]))
                
                if np.sum(mask_nC) > min_cluster_size:
                    xy_nC = xy_list[mask_nC,:]
                    label_nC = label_temp[mask_nC]

                    xy_proj = (xy_nC @ eigvec_all[nC,:].reshape(-1, 1)).flatten()/np.sum(eigvec_all[nC,:]**2)
                    xy_proj = (xy_proj - np.mean(xy_proj))/np.std(xy_proj)
                    
                    statistic, pval = diptest.diptest(xy_proj)
                    pval_save[nC] = pval
                    
                    if (pval <= p_crit and np.sum(label_temp==2*nC) > min_cluster_size and np.sum(label_temp==2*nC+1) > min_cluster_size)\
                       or (pval <= 1-p_crit and np.sum(mask_nC) > max_cluster_size):
                        if centroid is None:
                            centroid = centroid_temp[2*nC:2*(nC+1),:].copy()
                        else:
                            centroid = np.vstack((centroid, centroid_temp[2*nC:2*(nC+1),:]))
                        label[mask_nC] = ind_label + label_nC - np.min(label_nC).astype(int)
                        ind_label += 2
                        split_check = True
                    else:
                        if centroid is None:
                            centroid = centroid_prev[nC,:].reshape(1, -1)
                        else:
                            centroid = np.vstack((centroid, centroid_prev[nC,:]))
                        label[mask_nC] = ind_label
                        ind_label += 1

            centroid, label = weighted_kmeans(xy_list, np.ones(xy_list.shape[0]), centroid.shape[0], maxiter_kmeans, centroid=centroid)
            
            fm = fowlkes_mallows_index(label_prev, label)
            
            if comm.rank == 0:
                np.savez(directory + "/data/debug_gmeans_" + savefilename + str(cnt) + "_rank" + str(comm.rank + self.rank_start), centroid=centroid, label=label, eigval=eigval, eigvec=eigvec,
                         centroid_temp=centroid_temp, centroid_prev=centroid_prev, xy_proj=xy_proj, pval=pval_save, label_prev=label_prev, ind_label=ind_label, xy_list=xy_list,
                         label_temp=label_temp, xy_nC=xy_nC, label_nC=label_nC, max_cluster_size=max_cluster_size, min_cluster_size=min_cluster_size)
                
            cnt += 1
            
            if not split_check or cnt > maxiter_gmeans or fm == 1:
                break
        
        if centroid.shape[0] > 1:
            while True:
                pval_sweep = np.zeros((centroid.shape[0], centroid.shape[0]))
                for nC1 in range(centroid.shape[0]):
                    for nC2 in range(nC1, centroid.shape[0]):
                        if nC1 != nC2:
                            mask_nC = np.isin(label, np.array([nC1,nC2]))
                            
                            xy_nC = xy_list[mask_nC,:]
                            label_nC = label[mask_nC]
                            eigval, eigvec = pca(xy_nC)
        
                            xy_proj = (xy_nC @ eigvec.reshape(-1, 1)).flatten()/np.sum(eigvec**2)
                            xy_proj = (xy_proj - np.mean(xy_proj))/np.std(xy_proj)
                            
                            statistic, pval_sweep[nC1,nC2] = diptest.diptest(xy_proj)
                
                centroid_merged = None
                label_merged = np.zeros(xy_list.shape[0])
                ind_merged = np.zeros(centroid.shape[0])
                merge_check = False
                if np.max(pval_sweep) > p_crit:
                    nC1, nC2 = np.argwhere(pval_sweep==np.max(pval_sweep))[0]
                    merge_check = True
                    
                    mask_nC = np.isin(label, np.array([nC1,nC2]))
                    xy_nC = xy_list[mask_nC,:]
                    
                    centroid_new = np.mean(xy_nC, axis=0)
                    centroid_merged = np.vstack((centroid[:nC1,:], centroid[nC1+1:nC2,:], centroid[nC2+1:,:], centroid_new))
                    
                    for nC in range(centroid.shape[0]):
                        if nC < nC1:
                            label_merged[label==nC] = nC
                        elif nC > nC1 and nC < nC2:
                            label_merged[label==nC] = nC - 1
                        elif nC > nC2:
                            label_merged[label==nC] = nC - 2
                    label_merged[mask_nC] = np.max(label_merged) + 1
                else:
                    centroid_merged = centroid.copy()
                    label_merged = label.copy()
                
                if len(centroid_merged.shape) == 1:
                    centroid_merged = centroid_merged.reshape(1, -1)
            
                centroid = centroid_merged.copy()
                label = label_merged.copy()

                if not merge_check:
                    break
        
        if comm.rank == 0:
            np.savez(directory + "/data/debug_gmeans_" + savefilename + "_final_rank" + str(comm.rank + self.rank_start), centroid=centroid, label=label, eigval=eigval, eigvec=eigvec,
                     centroid_temp=centroid_temp, centroid_prev=centroid_prev, xy_proj=xy_proj, pval=pval_save, label_prev=label_prev, ind_label=ind_label, xy_list=xy_list,
                     label_temp=label_temp, xy_nC=xy_nC, label_nC=label_nC, max_cluster_size=max_cluster_size, min_cluster_size=min_cluster_size)
        
        return centroid, label
    
    def generate_energy_distribution_data(self, Nruns_per_energy):
        if os.path.exists(directory + "/data/energy_distribution_" + str(comm.rank + self.rank_start) + ".npz"):
            os.remove(directory + "/data/energy_distribution_" + str(comm.rank + self.rank_start) + ".npz")
        
        source_xy = np.zeros(2)
        
        quo, rem = divmod(self.energy_bins, comm.size)
        data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
        data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
        energy_per_proc = self.energy_list[data_disp[comm.rank]:data_disp[comm.rank+1]]
        
        if self.RGB_scintillator:
            energy_distribution_per_proc = np.zeros((3, self.d_mean_bins, self.energy_bins))
            n_photons_per_proc = np.zeros((3, self.n_ph_bins, self.energy_bins))
        else:
            energy_distribution_per_proc = np.zeros((self.d_mean_bins, self.energy_bins))
            n_photons_per_proc = np.zeros((self.n_ph_bins, self.energy_bins))
        
        for nd in range(data_size[comm.rank]):
            self.make_simulation_macro(Nruns_per_energy, source_energy=energy_per_proc[nd], source_xy='center')
            wvl_dict, xy_dict, _, _ = self.read_output(Nruns_per_energy)
            
            for nr in range(Nruns_per_energy):
                if wvl_dict[nr].size > 3:
                    
                    ind_d_mean, d_mean = self.data_interpretation(xy_dict[nr])

                    if self.RGB_scintillator:
                        RGB_list = self.RGB_readings(wvl_dict[nr])
                        ind_RGB = np.argmax(np.sum(RGB_list, axis=0))
                        energy_distribution_per_proc[ind_RGB,int(ind_d_mean),int(data_disp[comm.rank]+nd)] += 1
                        
                        ind_n_ph = np.argmin(np.abs(wvl_dict[nr].size - self.n_ph_list))
                        n_photons_per_proc[ind_RGB,int(ind_n_ph),int(data_disp[comm.rank]+nd)] += 1
                    else:
                        energy_distribution_per_proc[int(ind_d_mean),int(data_disp[comm.rank]+nd)] += 1
                        
                        ind_n_ph = np.argmin(np.abs(wvl_dict[nr].size - self.n_ph_list))
                        n_photons_per_proc[int(ind_n_ph),int(data_disp[comm.rank]+nd)] += 1
        
        np.savez(directory + "/data/energy_distribution_" + str(comm.rank + self.rank_start), energy_distribution=energy_distribution_per_proc, energy_list=energy_per_proc,
                 n_photons_per_proc=n_photons_per_proc)
        
        while True:
            filecheck = np.zeros(comm.size)
            for nproc in range(self.rank_start, comm.size+self.rank_start):
                filecheck[nproc-self.rank_start] = os.path.exists(directory + "/data/energy_distribution_" + str(nproc) + ".npz")
            if np.sum(filecheck) == comm.size:
                break
            time.sleep(1)
        
        time.sleep(5)
        
        if self.energy_distribution is None:
            if self.RGB_scintillator:
                self.energy_distribution = np.zeros((3, self.d_mean_bins, self.energy_bins))
                self.n_photon_distribution = np.zeros((3, self.n_ph_bins, self.energy_bins))
            else:
                self.energy_distribution = np.zeros((self.d_mean_bins, self.energy_bins))
                self.n_photon_distribution = np.zeros((self.n_ph_bins, self.energy_bins))
        
        for nproc in range(self.rank_start, comm.size+self.rank_start):
            data_proc = np.load(directory + "/data/energy_distribution_" + str(nproc) + ".npz")
            self.energy_distribution += data_proc['energy_distribution']
            self.n_photon_distribution += data_proc['n_photons_per_proc']
        
        if comm.rank == 0:
            np.savez(directory + "/data/energy_distribution_data_" + self.distribution_datafile, energy_distribution=self.energy_distribution, n_photon_distribution=self.n_photon_distribution)
        
        comm.Barrier()
        if comm.rank == 0:
            sync_check = 1
        else:
            sync_check = None
        sync_check = comm.bcast(sync_check, root=0)
    
    def test_energy_reconstruction(self, Nimages):
        if self.verbose:
            print('\n### Energy Reconstruction Test', flush=True)
        
        # Run Geant4 Simulation & Scatter Data to Processes
        quo, rem = divmod(Nimages, comm.size)
        data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
        data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])
        
        time.sleep(comm.rank)
        
        t1 = time.time()
        self.make_simulation_macro(data_size[comm.rank], source_xy='center')
        wvl_dict, xy_dict, energySrc_list, xySrc_list = self.read_output(data_size[comm.rank])
        t2 = time.time()
        if self.verbose:
            print('    Geant4: ' + str(t2-t1), flush=True)
    
        # Initialize Metric Arrays
        Mconf_per_proc = np.zeros((self.energy_tgt.size-1, self.energy_tgt.size-1))
        nPhoton_per_proc = np.zeros(0)
        
        # Normalize Energy Distribution Probability
        if self.RGB_scintillator:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=(0,1))[np.newaxis,np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=2)[:,:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_distribution.shape[1], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,:,i_tgt] = np.sum(energy_distribution_temp[:,:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=2)
        else:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=0)[np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=1)[:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,i_tgt] = np.sum(energy_distribution_temp[:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=1)
        
        # Compute Confusion Matrix
        if self.verbose:
            print('    ', end='', flush=True)
        for ni in range(data_size[comm.rank]):
            if verbose:
                if ni%(data_size[comm.rank]/10) == 0:
                    print(int(ni//(data_size[comm.rank]/100)), end='', flush=True)
                elif ni%(data_size[comm.rank]/100) == 0:
                    print('/', end='', flush=True)
            
            if wvl_dict[ni].size > 10:
                ind_d_mean, d_mean = self.data_interpretation(xy_dict[ni])
                
                if self.RGB_scintillator:
                    RGB_list = self.RGB_readings(wvl_dict[ni])
                    ind_RGB = np.argmax(np.mean(RGB_list, axis=0))
                    xrayInc = energy_distribution_norm[int(ind_RGB),ind_d_mean,:]
                else:
                    xrayInc = energy_distribution_norm[ind_d_mean,:]
                    
                for ntgt in range(self.energy_tgt.size-1):
                    if energySrc_list[ni] >= self.energy_tgt[ntgt] and energySrc_list[ni] < self.energy_tgt[ntgt+1]:
                        break
                        
                Mconf_per_proc[ntgt,:] += xrayInc
                nPhoton_per_proc = np.append(nPhoton_per_proc, wvl_dict[ni].size)
        if self.verbose:
            print('', flush=True)
    
        # Save Results per Process
        if not os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton"):
            os.mkdir(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton")
        if not os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf"):
            os.mkdir(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf")
        if comm.rank != 0:
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf/rank" + str(comm.rank + self.rank_start),
                     Mconf_per_proc=Mconf_per_proc)
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton/rank" + str(comm.rank + self.rank_start),
                     nPhoton_per_proc=nPhoton_per_proc)
    
        # Collect Results
        Mconf = Mconf_per_proc.copy()
        nPhoton = nPhoton_per_proc.copy()
    
        if comm.rank == 0:
            for nr in range(1+self.rank_start, comm.size+self.rank_start):
                while True:
                    if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton/rank" + str(nr) + ".npz"):
                        try:
                            with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton/rank" + str(nr) + ".npz") as data:
                                nPhoton = np.hstack((nPhoton, data['nPhoton_per_proc']))
                            break
                        except:
                            pass
                    else:
                        time.sleep(1)
                os.remove(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton/rank" + str(nr) + ".npz")
                
                while True:
                    if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf/rank" + str(nr) + ".npz"):
                        try:
                            with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf/rank" + str(nr) + ".npz") as data:
                                Mconf += data['Mconf_per_proc']
                            break
                        except:
                            pass
                    else:
                        time.sleep(1)
                os.remove(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf/rank" + str(nr) + ".npz")
    
        comm.Barrier()
    
        Mconf_norm = Mconf/np.sum(Mconf, axis=1)[:,np.newaxis]
        accuracy = np.trace(Mconf_norm)/np.sum(Mconf_norm)
    
        if comm.rank == 0:
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf_nPhoton_" + str(int(self.energy_tgt[1])) + "_" + str(int(self.energy_tgt[2])), nPhoton=nPhoton, Mconf=Mconf)
        
        comm.Barrier()
        
        return accuracy
    
    def test_spatial_reconstruction(self, Nphoton_rate, Nimages):
        if self.verbose:
            print('\n### Spatial Reconstruction Test | Photon Rate: ' + str(Nphoton_rate), flush=True)
        
        # Run Geant4 Simulation & Scatter Data to Processes
        quo, rem = divmod(Nimages, comm.size)
        data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
        data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])
        
        time.sleep(comm.rank)
        
        t1 = time.time()
        self.make_simulation_macro(Nphoton_rate*data_size[comm.rank])
        wvl_dict, xy_dict, energySrc_list, xySrc_list = self.read_output(Nphoton_rate*data_size[comm.rank])
        t2 = time.time()
        if self.verbose:
            print('    Geant4: ' + str(t2-t1), flush=True)
        
        # Initialize Metric Arrays
        fowlkes_mallows_index_per_proc = np.zeros(data_size[comm.rank])
        
        # Normalize Energy Distribution Probability
        if self.RGB_scintillator:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=(0,1))[np.newaxis,np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=2)[:,:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_distribution.shape[1], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,:,i_tgt] = np.sum(energy_distribution_temp[:,:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=2)
        else:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=0)[np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=1)[:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,i_tgt] = np.sum(energy_distribution_temp[:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=1)
        
        for ni in range(data_size[comm.rank]):
            wvl_list = np.zeros(0)
            x_list = np.zeros(0)
            y_list = np.zeros(0)
            event_label = np.zeros(0)
            source_energy_list = np.zeros(Nphoton_rate)
            source_xy_list = np.zeros((Nphoton_rate, 2))
            
            # Truncate Data per Image
            for nph in range(Nphoton_rate):
                if energySrc_list[ni*Nphoton_rate+nph] >= self.energy_tgt[0] and energySrc_list[ni*Nphoton_rate+nph] <= self.energy_tgt[-1] and wvl_dict[ni*Nphoton_rate+nph].size > 0:
                    source_energy_list[nph] = energySrc_list[ni*Nphoton_rate+nph]
                    source_xy_list[nph,:] = xySrc_list[ni*Nphoton_rate+nph,:] # mm
                    
                    wvl_list = np.hstack((wvl_list, wvl_dict[ni*Nphoton_rate+nph]))
                    x_list = np.hstack((x_list, xy_dict[ni*Nphoton_rate+nph][:,0]))
                    y_list = np.hstack((y_list, xy_dict[ni*Nphoton_rate+nph][:,1]))
                    event_label = np.hstack((event_label, nph*np.ones(wvl_dict[ni*Nphoton_rate+nph].size)))
            
            xy_list = np.vstack((x_list, y_list)).T
            
            # Confirm that there are Enough Photons per Channel
            if self.RGB_scintillator:
                RGB_list = self.RGB_readings(wvl_list)
                nPhoton = np.zeros(3)
                for i in range(3):
                    nPhoton[i] = np.sum(np.argmax(RGB_list, axis=1)==i)
                nPhoton_test = np.any(nPhoton>10)
                if self.verbose:
                    print('    nPhoton: ' + str(nPhoton[0]) + ' | ' + str(nPhoton[1]) + ' | ' + str(nPhoton[2]), flush=True)
            else:
                RGB_list = None
                nPhoton_test = wvl_list.size > 10
            
            # Clustering
            if nPhoton_test:
                t1 = time.time()
                fowlkes_mallows_index_per_proc[ni] = self.reconstruct_image(energy_distribution_norm, source_energy_list, source_xy_list, xy_list, RGB_list,
                                                                            event_label, Nphoton_rate)
                t2 = time.time()
                if self.verbose:
                    print('    Reconstruction: ' + str(t2-t1), flush=True)
            else:
                fowlkes_mallows_index_per_proc[ni] = np.nan
        
        # Collect Data from Processes
        fowlkes_mallows_index = np.zeros(Nimages)
        comm.Gatherv(fowlkes_mallows_index_per_proc, [fowlkes_mallows_index, data_size, data_disp, MPI.DOUBLE], root=0)

        if comm.rank == 0:
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/fowlkes_mallows_pRate_" + str(Nphoton_rate), fowlkes_mallows_index=fowlkes_mallows_index)
        
        comm.Barrier()

    def compute_SNR(self, Nphoton_rate, Nimages):
        if self.verbose:
            print('\n### Signal-to-Noise Ratio Test | Frames: ' + str(Nimages), flush=True)
        
        # Run Geant4 Simulation & Scatter Data to Processes
        quo, rem = divmod(Nimages, comm.size)
        data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
        data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])
        
        time.sleep(comm.rank)
        
        t1 = time.time()
        self.make_simulation_macro(Nphoton_rate*data_size[comm.rank])
        wvl_dict, xy_dict, energySrc_list, xySrc_list = self.read_output(Nphoton_rate*data_size[comm.rank])
        t2 = time.time()
        if self.verbose:
            print('    Geant4: ' + str(t2-t1), flush=True)
        
        # Normalize Energy Distribution Probability
        if self.RGB_scintillator:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=(0,1))[np.newaxis,np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=2)[:,:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_distribution.shape[1], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,:,i_tgt] = np.sum(energy_distribution_temp[:,:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=2)
        else:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=0)[np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=1)[:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,i_tgt] = np.sum(energy_distribution_temp[:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=1)
        
        for ni in range(data_size[comm.rank]):
            wvl_list = np.zeros(0)
            x_list = np.zeros(0)
            y_list = np.zeros(0)
            event_label = np.zeros(0)
            source_energy_list = np.zeros(Nphoton_rate)
            source_xy_list = np.zeros((Nphoton_rate, 2))
            
            # Truncate Data per Image
            for nph in range(Nphoton_rate):
                if energySrc_list[ni*Nphoton_rate+nph] >= self.energy_tgt[0] and energySrc_list[ni*Nphoton_rate+nph] <= self.energy_tgt[-1] and wvl_dict[ni*Nphoton_rate+nph].size > 0:
                    source_energy_list[nph] = energySrc_list[ni*Nphoton_rate+nph]
                    source_xy_list[nph,:] = xySrc_list[ni*Nphoton_rate+nph,:] # mm
                    
                    wvl_list = np.hstack((wvl_list, wvl_dict[ni*Nphoton_rate+nph]))
                    x_list = np.hstack((x_list, xy_dict[ni*Nphoton_rate+nph][:,0]))
                    y_list = np.hstack((y_list, xy_dict[ni*Nphoton_rate+nph][:,1]))
                    event_label = np.hstack((event_label, nph*np.ones(wvl_dict[ni*Nphoton_rate+nph].size)))
            
            xy_list = np.vstack((x_list, y_list)).T
            
            # Confirm that there are Enough Photons per Channel
            if self.RGB_scintillator:
                RGB_list = self.RGB_readings(wvl_list)
                nPhoton = np.zeros(3)
                for i in range(3):
                    nPhoton[i] = np.sum(np.argmax(RGB_list, axis=1)==i)
                nPhoton_test = np.any(nPhoton>10)
                if self.verbose:
                    print('    nPhoton: ' + str(nPhoton[0]) + ' | ' + str(nPhoton[1]) + ' | ' + str(nPhoton[2]), flush=True)
            else:
                RGB_list = None
                nPhoton_test = wvl_list.size > 10
            
            # Clustering
            if nPhoton_test:
                t1 = time.time()
                fm_temp = self.reconstruct_image(energy_distribution_norm, source_energy_list, source_xy_list, xy_list, RGB_list,
                                                 event_label, Nphoton_rate)
                t2 = time.time()
                if self.verbose:
                    print('    Reconstruction: ' + str(t2-t1), flush=True)
        
        if self.verbose:
            print('\n    Image Gathering Progress:', flush=True)
            print('    ', end='', flush=True)
    
        if not os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_Nimg" + str(Nimages)):
            os.mkdir(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_Nimg" + str(Nimages))
        if comm.rank != 0:
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_Nimg" + str(Nimages) + "/rank" + str(comm.rank + self.rank_start),
                     image_true=self.image_true, image_reconstr=self.image_reconstr)
    
        if comm.rank == 0:
            for nr in range(1+self.rank_start, comm.size+self.rank_start):
                if nr%100 == 0:
                    print('+', end='', flush=True)
                elif nr%10 == 0:
                    print('/', end='', flush=True)
                else:
                    print('|', end='', flush=True)
                while True:
                    if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_Nimg" + str(Nimages) + "/rank" + str(nr) + ".npz"):
                        try:
                            with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_Nimg" + str(Nimages) + "/rank" + str(nr) + ".npz") as data:
                                self.image_true += data['image_true']
                                self.image_reconstr += data['image_reconstr']
                            break
                        except:
                            pass
                    else:
                        time.sleep(1)
                os.remove(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_Nimg" + str(Nimages) + "/rank" + str(nr) + ".npz")
            print('', flush=True)

        if self.RGB_scintillator:
            image_reconstr_temp = np.sum(self.image_reconstr, axis=2)
        else:
            image_reconstr_temp = self.image_reconstr.copy()
        SNR = np.mean(image_reconstr_temp)/np.std(image_reconstr_temp)

        comm.Barrier()
        
        return SNR
    
    def test_phantom(self, Nphoton_rate, Nimages):
        if self.verbose:
            print('\n### Reconstructing Phantom Images | Photon Rate: ' + str(Nphoton_rate), flush=True)
        
        if comm.rank == 0:
            phantom.linAttCoeff_contrast_dat("AM", self.energy_list, self.IConc, 'IBlood', self.G4directory)
            phantom.linAttCoeff_contrast_dat("AM", self.energy_list, self.GdConc, 'GdBlood', self.G4directory)
        
        # Run Geant4 Simulation & Scatter Data to Processes
        quo, rem = divmod(Nimages, comm.size)
        data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
        data_disp = np.array([sum(data_size[:p]) for p in range(comm.size)])
        
        time.sleep(comm.rank)
        
        t1 = time.time()
        self.make_simulation_macro(Nphoton_rate*data_size[comm.rank])
        wvl_dict, xy_dict, energySrc_list, xySrc_list = self.read_output(Nphoton_rate*data_size[comm.rank])
        t2 = time.time()
        if self.verbose:
            print('    Geant4: ' + str(t2-t1), flush=True)
        
        # Normalize Energy Distribution Probability
        if self.RGB_scintillator:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=(0,1))[np.newaxis,np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=2)[:,:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_distribution.shape[1], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,:,i_tgt] = np.sum(energy_distribution_temp[:,:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=2)
        else:
            energy_distribution_temp = self.energy_distribution + 1e-8
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=0)[np.newaxis,:]
            energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=1)[:,np.newaxis]
            energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_tgt.size-1))
            for i_tgt in range(self.energy_tgt.size-1):
                energy_distribution_norm[:,i_tgt] = np.sum(energy_distribution_temp[:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=1)
        
        if self.verbose:
            print('    Reconstruction Progress:', flush=True)
            print('    ', end='', flush=True)
            for i in range(100):
                if i%10 == 0:
                    print(int(i/10), end='', flush=True)
                else:
                    print('-', end='', flush=True)
            print('', flush=True)
            print('    ', end='', flush=True)
        
        for ni in range(data_size[comm.rank]):
            if self.verbose:
                if ni%(int(data_size[comm.rank]/100)) == 0:
                    print('|', end='', flush=True)
        
            wvl_list = np.zeros(0)
            x_list = np.zeros(0)
            y_list = np.zeros(0)
            event_label = np.zeros(0)
            source_energy_list = np.zeros(Nphoton_rate)
            source_xy_list = np.zeros((Nphoton_rate, 2))
        
            # Construct Ground Truth Image
            t1 = time.time()
            for nph in range(Nphoton_rate):
                if energySrc_list[ni*Nphoton_rate+nph] >= self.energy_tgt[0] and energySrc_list[ni*Nphoton_rate+nph] <= self.energy_tgt[-1] and wvl_dict[ni*Nphoton_rate+nph].size > 0:
                    source_energy_list[nph] = energySrc_list[ni*Nphoton_rate+nph]
                    source_xy_list[nph,:] = xySrc_list[ni*Nphoton_rate+nph,:] # mm
                    
                    ind_x0 = int((xySrc_list[ni*Nphoton_rate+nph,0] - self.dimDetVoxel[0]/2e3)//(self.dimDetVoxel[0]/1e3))
                    x0 = self.dimDetVoxel[0]/2e3 + ind_x0*self.dimDetVoxel[0]/1e3
                    ind_x1 = ind_x0 + 1
                    x1 = self.dimDetVoxel[0]/2e3 + ind_x1*self.dimDetVoxel[0]/1e3
                    if np.abs(x0 - xySrc_list[ni*Nphoton_rate+nph,0]) <= np.abs(x1 - xySrc_list[ni*Nphoton_rate+nph,0]):
                        source_x_ind = ind_x0
                    else:
                        source_x_ind = ind_x1
                        
                    ind_y0 = int((xySrc_list[ni*Nphoton_rate+nph,1] - self.dimDetVoxel[1]/2e3)//(self.dimDetVoxel[1]/1e3))
                    y0 = self.dimDetVoxel[1]/2e3 + ind_y0*self.dimDetVoxel[1]/1e3
                    ind_y1 = ind_y0 + 1
                    y1 = self.dimDetVoxel[1]/2e3 + ind_y1*self.dimDetVoxel[1]/1e3
                    if np.abs(y0 - xySrc_list[ni*Nphoton_rate+nph,1]) <= np.abs(y1 - xySrc_list[ni*Nphoton_rate+nph,1]):
                        source_y_ind = ind_y0
                    else:
                        source_y_ind = ind_y1
                    
                    for i_tgt in range(self.energy_tgt.size-1):
                        if energySrc_list[ni*Nphoton_rate+nph] >= self.energy_tgt[i_tgt] and energySrc_list[ni*Nphoton_rate+nph] < self.energy_tgt[i_tgt+1]:
                            source_energy_ind = i_tgt
                            break
            
                    self.image_true[source_x_ind,source_y_ind,source_energy_ind] += 1
        
                    wvl_list = np.hstack((wvl_list, wvl_dict[ni*Nphoton_rate+nph]))
                    x_list = np.hstack((x_list, xy_dict[ni*Nphoton_rate+nph][:,0]))
                    y_list = np.hstack((y_list, xy_dict[ni*Nphoton_rate+nph][:,1]))
                    event_label = np.hstack((event_label, nph*np.ones(wvl_dict[ni*Nphoton_rate+nph].size)))
            t2 = time.time()
            
            xy_list = np.vstack((x_list, y_list)).T
            
            # Confirm that there are Enough Photons per Channel
            if self.RGB_scintillator:
                RGB_list = self.RGB_readings(wvl_list)
                nPhoton = np.zeros(3)
                for i in range(3):
                    nPhoton[i] = np.sum(np.argmax(RGB_list, axis=1)==i)
                nPhoton_test = np.any(nPhoton>10)
            else:
                RGB_list = None
                nPhoton_test = wvl_list.size > 10
            
            # Clustering
            if nPhoton_test:
                t1 = time.time()
                fm_temp = self.reconstruct_image(energy_distribution_norm, source_energy_list, source_xy_list, xy_list, RGB_list,
                                                 event_label, Nphoton_rate)
                t2 = time.time()
        
        if self.verbose:
            print('\n    Image Gathering Progress:', flush=True)
            print('    ', end='', flush=True)
    
        if not os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate)):
            os.mkdir(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate))
        if comm.rank != 0:
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate) + "/rank" + str(comm.rank + self.rank_start),
                     image_true=self.image_true, image_reconstr=self.image_reconstr)
    
        if comm.rank == 0:
            for nr in range(1+self.rank_start, comm.size+self.rank_start):
                if nr%100 == 0:
                    print('+', end='', flush=True)
                elif nr%10 == 0:
                    print('/', end='', flush=True)
                else:
                    print('|', end='', flush=True)
                while True:
                    if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate) + "/rank" + str(nr) + ".npz"):
                        try:
                            with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate) + "/rank" + str(nr) + ".npz") as data:
                                self.image_true += data['image_true']
                                self.image_reconstr += data['image_reconstr']
                            break
                        except:
                            pass
                    else:
                        time.sleep(1)
                os.remove(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate) + "/rank" + str(nr) + ".npz")
            print('', flush=True)
            
            # Gmeans Reconstruction
            image_true_flat = self.compress_image(self.image_true, 10).reshape(-1, self.energy_tgt.size-1)
            offset = 1
            image_reconstr_flat = self.compress_image(self.image_reconstr, 10)[offset:-offset,offset:-offset,:].reshape(-1, self.energy_tgt.size-1)
            
            npts_true = image_true_flat.shape[0]
            npts_reconstr = image_reconstr_flat.shape[0]
            
            Bin_src = np.zeros(self.energy_tgt.size-1)
            for i_tgt in range(self.energy_tgt.size-1):
                Bin_src[i_tgt] = np.sum(self.xraySrc[self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]])
            Bin_src *= Nphoton_rate*Nimages
            
            mu_true = -(1/self.dimSample[2])*np.log(image_true_flat/Bin_src[np.newaxis,:]*npts_true)
            mu_reconstr = -(1/self.dimSample[2])*np.log(image_reconstr_flat/Bin_src[np.newaxis,:]*npts_reconstr)

            centroid_true, label_true = self.gmeans(mu_true, 20, 20, min_cluster_size=10, max_cluster_size=npts_true, savefilename='true')
            centroid_reconstr, label_reconstr_temp = self.gmeans(mu_reconstr, 20, 20, min_cluster_size=10, max_cluster_size=npts_reconstr, savefilename='reconstr')
            
            label_true = label_true.reshape(int(self.NDetVoxel[0]/10), int(self.NDetVoxel[1]/10))
            label_reconstr = (np.max(label_reconstr_temp) + 1)*np.ones((int(self.NDetVoxel[0]/10), int(self.NDetVoxel[1]/10)))
            label_reconstr[offset:-offset,offset:-offset] = label_reconstr_temp.reshape(int(self.NDetVoxel[0]/10)-2*offset, int(self.NDetVoxel[1]/10)-2*offset)
            
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate), image_true=self.image_true, image_reconstr=self.image_reconstr,
                     centroid_true=centroid_true, label_true=label_true, centroid_reconstr=centroid_reconstr, mu_true=mu_true, mu_reconstr=mu_reconstr, Bin_src=Bin_src,
                     label_reconstr=label_reconstr)
        
        comm.Barrier()
    
    def test_phantom_simple_model(self, Npixel_side, system):
        if self.RGB_scintillator:
            scint_emission = self.xraySrc.copy()#*np.sum(self.energy_distribution, axis=(0,1))/1e5
        else:
            scint_emission = self.xraySrc*np.sum(self.energy_distribution, axis=0)/1e5
        #scint_emission /= np.sum(scint_emission)
    
        if system == 'medical':
            p_crit = 0.05
        
            Npix_tot = Npixel_side**2
            IConc_list = np.array([self.IConc])
            GdConc_list = np.array([self.GdConc])
            boneprop = 0.051 #5cm 0.056 | 1cm 0.051
            muscleprop = 1.0 # 1.0/0.6963
            Iprop = 0.33 #0.29 | 0.33 | 1% 0.33
            Gdprop = 0.37 #0.49 | 0.53 | 1% 0.37
            noise_list = np.linspace(0.0, 0.01, 11)
            noise_list[0] = 0.001#0.001
        
            linAttCoeff_media = phantom.linAttCoeff_cylinder_phantom("AM", self.energy_list, IConc_list, GdConc_list, np.array([1,28,48]))
            #1 Mineral bone, 28 Muscle tissue, 48 Adipose tissue
            
            R_cyl = 0.175
            
            I_trans = np.zeros((Npix_tot, self.energy_bins))
            coord = np.linspace(0, 1, Npixel_side)
            X, Y = np.meshgrid(coord, coord)
            mask_tissue = np.ones(Npix_tot).astype(bool)
            
            label0 = np.zeros(Npix_tot)
            mask_bone = np.sqrt((X - 0.275)**2 + (Y - 0.275)**2).reshape(-1) <= R_cyl
            mask_tissue[mask_bone] = 0
            I_trans[mask_bone,:] = np.exp(-(linAttCoeff_media[0,:]*boneprop + linAttCoeff_media[2,:]*(1 - boneprop))*self.dimSample[2])*scint_emission
            label0[mask_bone] = 1
            
            mask_muscle = np.sqrt((X - 0.725)**2 + (Y - 0.275)**2).reshape(-1) <= R_cyl
            mask_tissue[mask_muscle] = 0
            I_trans[mask_muscle,:] = np.exp(-linAttCoeff_media[1,:]*self.dimSample[2])*scint_emission
            label0[mask_muscle] = 2
            
            mask_IBlood = np.sqrt((X - 0.275)**2 + (Y - 0.725)**2).reshape(-1) <= R_cyl
            mask_tissue[mask_IBlood] = 0
            I_trans[mask_IBlood,:] = np.exp(-(linAttCoeff_media[3,:]*Iprop + linAttCoeff_media[2,:]*(1 - Iprop))*self.dimSample[2])*scint_emission
            label0[mask_IBlood] = 3
            
            mask_GdBlood = np.sqrt((X - 0.725)**2 + (Y - 0.725)**2).reshape(-1) <= R_cyl
            mask_tissue[mask_GdBlood] = 0
            I_trans[mask_GdBlood,:] = np.exp(-(linAttCoeff_media[4,:]*Gdprop + linAttCoeff_media[2,:]*(1 - Gdprop))*self.dimSample[2])*scint_emission
            label0[mask_GdBlood] = 4
            
            I_trans[mask_tissue,:] = np.exp(-linAttCoeff_media[2,:]*self.dimSample[2])*scint_emission
            label0[mask_tissue] = 0
        
        elif system == 'battery': # 86 um
            Npix_tot = int(2*Npixel_side**2/3)
            impurity = 1/(15 + 70 + 1)
            Cuprop = 0.49
            Niprop = 0.5
            Pbprop = 0.135
            Feprop = 0.71
            Crprop = 1.0
            Znprop = 0.57
            Alprop = 15/(15 + 70 + 1)
            NMCprop = 70/(15 + 70 + 1)
            noise_list = np.linspace(0.0, 0.01, 11)
            noise_list[0] = 0.001
            
            linAttCoeff_media = phantom.linAttCoeff_cylinder_phantom_battery(self.energy_list)
            
            R_cyl = 0.125
            
            I_trans = np.zeros((Npix_tot, self.energy_bins))
            coordX = np.linspace(0, 1, Npixel_side)
            coordY = np.linspace(0, 2/3, int(2*Npixel_side/3))
            X, Y = np.meshgrid(coordX, coordY, indexing='ij')
            mask_NMC = np.ones(Npix_tot).astype(bool)
            
            label0 = np.zeros(Npix_tot)
            mask_Cu = np.sqrt((X - 0.1875)**2 + (Y - 0.275*2/3)**2).reshape(-1) <= R_cyl
            mask_NMC[mask_Cu] = 0
            I_trans[mask_Cu,:] = np.exp(-(linAttCoeff_media[2,:]*Cuprop*impurity + linAttCoeff_media[8,:]*(1 - Cuprop)*impurity + linAttCoeff_media[0,:]*Alprop\
                                        + linAttCoeff_media[1,:]*NMCprop)*self.dimSample[2])*self.xraySrc
            label0[mask_Cu] = 1
            
            mask_Ni = np.sqrt((X - 0.5)**2 + (Y - 0.275*2/3)**2).reshape(-1) <= R_cyl
            mask_NMC[mask_Ni] = 0
            I_trans[mask_Ni,:] = np.exp(-(linAttCoeff_media[3,:]*Niprop*impurity + linAttCoeff_media[8,:]*(1 - Niprop)*impurity + linAttCoeff_media[0,:]*Alprop\
                                        + linAttCoeff_media[1,:]*NMCprop)*self.dimSample[2])*self.xraySrc
            label0[mask_Ni] = 2
            
            mask_Pb = np.sqrt((X - 0.8125)**2 + (Y - 0.275*2/3)**2).reshape(-1) <= R_cyl
            mask_NMC[mask_Pb] = 0
            I_trans[mask_Pb,:] = np.exp(-(linAttCoeff_media[4,:]*Pbprop*impurity + linAttCoeff_media[8,:]*(1 - Pbprop)*impurity + linAttCoeff_media[0,:]*Alprop\
                                        + linAttCoeff_media[1,:]*NMCprop)*self.dimSample[2])*self.xraySrc
            label0[mask_Pb] = 3
            
            mask_Fe = np.sqrt((X - 0.1875)**2 + (Y - 0.725*2/3)**2).reshape(-1) <= R_cyl
            mask_NMC[mask_Fe] = 0
            I_trans[mask_Fe,:] = np.exp(-(linAttCoeff_media[5,:]*Feprop*impurity + linAttCoeff_media[8,:]*(1 - Feprop)*impurity + linAttCoeff_media[0,:]*Alprop\
                                        + linAttCoeff_media[1,:]*NMCprop)*self.dimSample[2])*self.xraySrc
            label0[mask_Fe] = 4
            
            mask_Cr = np.sqrt((X - 0.5)**2 + (Y - 0.725*2/3)**2).reshape(-1) <= R_cyl
            mask_NMC[mask_Cr] = 0
            I_trans[mask_Cr,:] = np.exp(-(linAttCoeff_media[6,:]*Crprop*impurity + linAttCoeff_media[8,:]*(1 - Crprop)*impurity + linAttCoeff_media[0,:]*Alprop\
                                        + linAttCoeff_media[1,:]*NMCprop)*self.dimSample[2])*self.xraySrc
            label0[mask_Cr] = 5
            
            mask_Zn = np.sqrt((X - 0.8125)**2 + (Y - 0.725*2/3)**2).reshape(-1) <= R_cyl
            mask_NMC[mask_Zn] = 0
            I_trans[mask_Zn,:] = np.exp(-(linAttCoeff_media[7,:]*Znprop*impurity + linAttCoeff_media[8,:]*(1 - Znprop)*impurity + linAttCoeff_media[0,:]*Alprop\
                                          + linAttCoeff_media[1,:]*NMCprop)*self.dimSample[2])*self.xraySrc
            label0[mask_Zn] = 6
            
            I_trans[mask_NMC,:] = np.exp(-(linAttCoeff_media[8,:]*impurity + linAttCoeff_media[0,:]*Alprop + linAttCoeff_media[1,:]*NMCprop)*self.dimSample[2])*self.xraySrc
            label0[mask_NMC] = 0
        
        elif system == 'baggage': # 1 cm
            p_crit = 0.05
        
            Npix_tot = int(2*Npixel_side**2/5)
            PETprop = 0.645 #0.645/0.317 0.65/0.314 0.67/0.304
            HDPEprop = 1.0 #1.0/0.301
            muscleprop = 0.685 #0.68/0.333 0.72/0.315 0.76/0.298
            H2Oprop = 0.725 #0.725/0.329 0.77/0.310 0.79/0.301
            celluloseprop = 0.555 #0.555/0.320 0.58/0.305
            SiO2prop = 0.157 #0.155/0.401 0.18/0.360 0.22/0.306
            ceramicprop = 0.134 #0.132/0.399 0.18/0.314 0.185/0.306
            Alprop = 0.1007 #0.0995/0.416 0.12/0.367 0.14/0.327 0.15/
            Feprop = 0.0051 #0.005/0.456 0.01/0.296
            Auprop = 0.000617 #0.00061/0.441 0.001/0.323 0.0011/0.300
            noise_list = np.linspace(0.0, 0.01, 1)
            noise_list[0] = 0.001
            
            linAttCoeff_media = phantom.linAttCoeff_cylinder_phantom_baggage(self.energy_list)
            
            R_cyl = 0.075
            
            I_trans = np.zeros((Npix_tot, self.energy_bins))
            coordX = np.linspace(0, 1, Npixel_side)
            coordY = np.linspace(0, 2/5, int(2*Npixel_side/5))
            X, Y = np.meshgrid(coordX, coordY, indexing='ij')
            mask_air = np.ones(Npix_tot).astype(bool)
            
            label0 = np.zeros(Npix_tot)
            mask_PET = np.sqrt((X - 0.25/6 - 0.075)**2 + (Y - 0.275*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_PET] = 0
            I_trans[mask_PET,:] = np.exp(-(linAttCoeff_media[1,:]*PETprop + linAttCoeff_media[0,:]*(1 - PETprop))*self.dimSample[2])*scint_emission
            label0[mask_PET] = 1
            
            mask_HDPE = np.sqrt((X - 0.25/6*2 - 0.075*3)**2 + (Y - 0.275*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_HDPE] = 0
            I_trans[mask_HDPE,:] = np.exp(-(linAttCoeff_media[2,:]*HDPEprop + linAttCoeff_media[0,:]*(1 - HDPEprop))*self.dimSample[2])*scint_emission
            label0[mask_HDPE] = 2
            
            mask_muscle = np.sqrt((X - 0.25/6*3 - 0.075*5)**2 + (Y - 0.275*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_muscle] = 0
            I_trans[mask_muscle,:] = np.exp(-(linAttCoeff_media[3,:]*muscleprop + linAttCoeff_media[0,:]*(1 - muscleprop))*self.dimSample[2])*scint_emission
            label0[mask_muscle] = 3
            
            mask_H2O = np.sqrt((X - 0.25/6*4 - 0.075*7)**2 + (Y - 0.275*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_H2O] = 0
            I_trans[mask_H2O,:] = np.exp(-(linAttCoeff_media[4,:]*H2Oprop + linAttCoeff_media[0,:]*(1 - H2Oprop))*self.dimSample[2])*scint_emission
            label0[mask_H2O] = 4
            
            mask_cellulose = np.sqrt((X - 0.25/6*5 - 0.075*9)**2 + (Y - 0.275*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_cellulose] = 0
            I_trans[mask_cellulose,:] = np.exp(-(linAttCoeff_media[5,:]*celluloseprop + linAttCoeff_media[0,:]*(1 - celluloseprop))*self.dimSample[2])*scint_emission
            label0[mask_cellulose] = 5
            
            mask_SiO2 = np.sqrt((X - 0.25/6 - 0.075)**2 + (Y - 0.725*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_SiO2] = 0
            I_trans[mask_SiO2,:] = np.exp(-(linAttCoeff_media[6,:]*SiO2prop + linAttCoeff_media[0,:]*(1 - SiO2prop))*self.dimSample[2])*scint_emission
            label0[mask_SiO2] = 6
            
            mask_ceramic = np.sqrt((X - 0.25/6*2 - 0.075*3)**2 + (Y - 0.725*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_ceramic] = 0
            I_trans[mask_ceramic,:] = np.exp(-(linAttCoeff_media[7,:]*ceramicprop + linAttCoeff_media[0,:]*(1 - ceramicprop))*self.dimSample[2])*scint_emission
            label0[mask_ceramic] = 7
            
            mask_Al = np.sqrt((X - 0.25/6*3 - 0.075*5)**2 + (Y - 0.725*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_Al] = 0
            I_trans[mask_Al,:] = np.exp(-(linAttCoeff_media[8,:]*Alprop + linAttCoeff_media[0,:]*(1 - Alprop))*self.dimSample[2])*scint_emission
            label0[mask_Al] = 8
            
            mask_Fe = np.sqrt((X - 0.25/6*4 - 0.075*7)**2 + (Y - 0.725*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_Fe] = 0
            I_trans[mask_Fe,:] = np.exp(-(linAttCoeff_media[9,:]*Feprop + linAttCoeff_media[0,:]*(1 - Feprop))*self.dimSample[2])*scint_emission
            label0[mask_Fe] = 9
            
            mask_Au = np.sqrt((X - 0.25/6*5 - 0.075*9)**2 + (Y - 0.725*2/5)**2).reshape(-1) <= R_cyl
            mask_air[mask_Au] = 0
            I_trans[mask_Au,:] = np.exp(-(linAttCoeff_media[10,:]*Auprop + linAttCoeff_media[0,:]*(1 - Auprop))*self.dimSample[2])*scint_emission
            label0[mask_Au] = 10
            
            I_trans[mask_air,:] = np.exp(-linAttCoeff_media[0,:]*self.dimSample[2])*scint_emission
            label0[mask_air] = 0
        
        # Incident to Detected Energy Mapping Matrix
        I_trans_sample = I_trans.copy()
        if self.RGB_scintillator:
            energy_distribution_temp = self.energy_distribution + 1e-8
            Pscint = energy_distribution_temp/np.sum(energy_distribution_temp, axis=(0,1))[np.newaxis,np.newaxis,:]
            Pdet = energy_distribution_temp/np.sum(energy_distribution_temp, axis=2)[:,:,np.newaxis]
            
            P = np.tensordot(Pscint.transpose(2,0,1), Pdet, axes=2)
        else:
            energy_distribution_temp = self.energy_distribution + 1e-8
            Pscint = energy_distribution_temp/np.sum(energy_distribution_temp, axis=0)[np.newaxis,:]
            Pdet = energy_distribution_temp/np.sum(energy_distribution_temp, axis=1)[:,np.newaxis]
        
            P = Pscint.T @ Pdet
            
        I_det = I_trans @ P
        
        Bin_trans = np.zeros((Npix_tot, self.energy_tgt.size-1))
        Bin_det = np.zeros((Npix_tot, self.energy_tgt.size-1))
        Bin_src = np.zeros(self.energy_tgt.size-1)
        for i_tgt in range(self.energy_tgt.size-1):
            Bin_trans[:,i_tgt] = np.sum(I_trans[:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=1)
            Bin_det[:,i_tgt] = np.sum(I_det[:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=1)
            
            Bin_src[i_tgt] = np.sum(scint_emission[self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]])
        
        fm_true = np.zeros(noise_list.size)
        fm_reconstr = np.zeros(noise_list.size)
        acc_true = np.zeros(noise_list.size)
        acc_reconstr = np.zeros(noise_list.size)
        nmat_true = np.zeros(noise_list.size)
        nmat_reconstr = np.zeros(noise_list.size)
        for nn in range(noise_list.size):
            print(nn)
            Bin_noise_trans = Bin_trans + np.mean(Bin_trans, axis=0)[np.newaxis,:]*noise_list[nn]*np.random.normal(size=(Npix_tot, self.energy_tgt.size-1))
            Bin_noise_det = Bin_det + np.mean(Bin_det, axis=0)[np.newaxis,:]*noise_list[nn]*np.random.normal(size=(Npix_tot, self.energy_tgt.size-1))
            Bin_noise_trans = np.maximum(Bin_noise_trans, 1e-6)
            Bin_noise_det = np.maximum(Bin_noise_det, 1e-6)
            
            mu_trans = -(1/self.dimSample[2])*np.log(Bin_noise_trans/Bin_src[np.newaxis,:])
            mu_det = -(1/self.dimSample[2])*np.log(Bin_noise_det/Bin_src[np.newaxis,:])
        
            npts_true = mu_trans.shape[0]
            centroid_true, label_true = self.gmeans(mu_trans, 20, 20, min_cluster_size=10, max_cluster_size=npts_true, p_crit=p_crit, savefilename='true')
            
            npts_reconstr = mu_det.shape[0]
            centroid_reconstr, label_reconstr = self.gmeans(mu_det, 20, 20, min_cluster_size=10, max_cluster_size=npts_reconstr, p_crit=p_crit, savefilename='reconstr')
            
            fm_true[nn] = fowlkes_mallows_index(label0, label_true)
            fm_reconstr[nn] = fowlkes_mallows_index(label0, label_reconstr)
            acc_true[nn] = label_accuracy(label0, label_true)
            acc_reconstr[nn] = label_accuracy(label0, label_reconstr)
            nmat_true[nn] = int(np.max(label_true)) + 1
            nmat_reconstr[nn] = int(np.max(label_reconstr)) + 1
        
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/all_data_simple_model_" + system + "_reconstruction_noise" + str(nn), mu_trans=mu_trans, mu_det=mu_det,
                     centroid_true=centroid_true, label_true=label_true, centroid_reconstr=centroid_reconstr, label_reconstr=label_reconstr, Bin_noise_trans=Bin_noise_trans,
                     Bin_noise_det=Bin_noise_det, linAttCoeff_media=linAttCoeff_media, I_trans=I_trans, I_trans_sample=I_trans_sample, P=P, I_det=I_det,
                     noise_list=noise_list, Bin_trans=Bin_trans, Bin_det=Bin_det, Bin_src=Bin_src, label0=label0, fm_true=fm_true, fm_reconstr=fm_reconstr, xraySrc=self.xraySrc,
                     energy_distribution_temp=energy_distribution_temp)
            
        np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/simple_model_" + system + "_reconstruction", linAttCoeff_media=linAttCoeff_media,
                 P=P, noise_list=noise_list, fm_true=fm_true, fm_reconstr=fm_reconstr, acc_true=acc_true, acc_reconstr=acc_reconstr, nmat_true=nmat_true, nmat_reconstr=nmat_reconstr)
    
    def test_phantom_CT(self, ind_slice, theta):
        mu_mapping, phantom_slice = phantom.phantom_CT("AM", ind_slice, self.energy_list, self.xraySrc, self.IConc, self.GdConc, self.G4directory)
        TrueImg = FBP.getTrueImg(mu_mapping, self.dimSampleVoxel[0], self.energy_tgt_ind, self.xraySrc)
        TrueImg = np.rot90(TrueImg, k=1, axes=(0,1))
        TrueImg_norm = (TrueImg - np.min(TrueImg))/(np.max(TrueImg) - np.min(TrueImg))
        TrueImg_reshape = TrueImg.reshape(-1, self.energy_tgt_ind.size-1)
        mask = np.abs(TrueImg_reshape[:,0] - 1) > 1e-6
        
        # Incident to Detected Energy Mapping Matrix
        if self.RGB_scintillator:
            energy_distribution_temp = self.energy_distribution + 1e-8
            Pscint = energy_distribution_temp/np.sum(energy_distribution_temp, axis=(0,1))[np.newaxis,np.newaxis,:]
            Pdet = Pscint/np.sum(Pscint, axis=2)[:,:,np.newaxis]
            
            P = np.tensordot(Pscint.transpose(2,0,1), Pdet, axes=2)
        else:
            energy_distribution_temp = self.energy_distribution + 1e-8
            Pscint = energy_distribution_temp/np.sum(energy_distribution_temp, axis=0)[np.newaxis,:]
            Pdet = Pscint/np.sum(Pscint, axis=1)[:,np.newaxis]
        
            P = Pscint.T @ Pdet
        
        noise_list = np.array([0.001,0.01,0.05,0.1])
        
        for nn in range(noise_list.size):
            sinogram_ideal = FBP.getProj_phantom(mu_mapping, self.dimSampleVoxel[0], theta, np.identity(self.energy_bins), self.energy_tgt_ind, self.xraySrc, noise_list[nn])
            sinogram = FBP.getProj_phantom(mu_mapping, self.dimSampleVoxel[0], theta, P, self.energy_tgt_ind, self.xraySrc, noise_list[nn])
            bpImg_ideal = np.zeros((phantom_slice.shape[0], phantom_slice.shape[1], self.energy_tgt_ind.size-1))
            bpImg = np.zeros((phantom_slice.shape[0], phantom_slice.shape[1], self.energy_tgt_ind.size-1))
            for nE in range(self.energy_tgt_ind.size-1):
                filtSino_ideal = FBP.projFilter(sinogram_ideal[:,:,nE])
                filtSino = FBP.projFilter(sinogram[:,:,nE])
                bpImg_ideal[:,:,nE] = FBP.backproject(filtSino_ideal, theta)
                bpImg[:,:,nE] = FBP.backproject(filtSino, theta)
            
#            bpImg_ideal_norm = (bpImg_ideal - np.min(bpImg_ideal))/(np.max(bpImg_ideal) - np.min(bpImg_ideal))
#            bpImg_norm = (bpImg - np.min(bpImg))/(np.max(bpImg) - np.min(bpImg))
#            
#            x0 = np.array([1.0,0.0])
#            for nE in range(self.energy_tgt_ind.size-1):
#                result = minimize(image_normalization_cost, x0, args=(TrueImg_norm[:,:,nE], bpImg_ideal_norm[:,:,nE]))
#                bpImg_ideal_norm = result.x[0]*bpImg_ideal_norm + result.x[1]
#                result = minimize(image_normalization_cost, x0, args=(TrueImg_norm[:,:,nE], bpImg_norm[:,:,nE]))
#                bpImg_norm = result.x[0]*bpImg_norm + result.x[1]
        
#            bpImg_ideal_norm = bpImg_ideal + 1e-8 - np.min(bpImg_ideal)
#            bpImg_ideal_norm = bpImg_ideal_norm.reshape(-1,self.energy_tgt_ind.size-1)
#            bpImg_ideal_norm = bpImg_ideal_norm/np.max(bpImg_ideal_norm)
#            bpImg_ideal_norm *= 1 - 1e-8
#            xyz_ideal = np.log(-np.log(bpImg_ideal_norm))
#            npts_ideal = xyz_ideal.shape[0]

            bpImg_ideal_reshape = bpImg_ideal.reshape(-1,self.energy_tgt_ind.size-1)[mask,:]
            npts_ideal = bpImg_ideal_reshape.shape[0]
            label_ideal = np.zeros(mask.size)
            centroid_ideal, label_ideal[mask] = self.gmeans(bpImg_ideal_reshape, 20, 20, min_cluster_size=10, max_cluster_size=npts_ideal, savefilename='ideal')
            
#            bpImg_norm = bpImg + 1e-8 - np.min(bpImg)
#            bpImg_norm = bpImg_norm.reshape(-1,self.energy_tgt_ind.size-1)
#            bpImg_norm = bpImg_norm/np.max(bpImg_norm)
#            bpImg_norm *= 1 - 1e-8
#            xyz_reconstr = np.log(-np.log(bpImg_norm))
#            npts_reconstr = xyz_reconstr.shape[0]
                
            bpImg_reshape = bpImg.reshape(-1,self.energy_tgt_ind.size-1)[mask,:]
            npts_reconstr = bpImg_reshape.shape[0]
            label_reconstr = np.zeros(mask.size)
            centroid_reconstr, label_reconstr[mask] = self.gmeans(bpImg_reshape, 20, 20, min_cluster_size=10, max_cluster_size=npts_reconstr, savefilename='reconstr')
        
            fm_ideal = fowlkes_mallows_index(phantom_slice.reshape(-1), label_ideal)
            fm_reconstr = fowlkes_mallows_index(phantom_slice.reshape(-1), label_reconstr)
        
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/CT_reconstruction_noise" + str(nn), mu_mapping=mu_mapping, TrueImg=TrueImg_norm, P=P, noise_list=noise_list,
                     phantom_slice=phantom_slice, centroid_ideal=centroid_ideal, label_ideal=label_ideal, centroid_reconstr=centroid_reconstr, label_reconstr=label_reconstr, fm_ideal=fm_ideal,
                     fm_reconstr=fm_reconstr, bpImg_ideal=bpImg_ideal, bpImg=bpImg)
    
    def compress_image(self, image, factor):
        image_new = np.sum(image.reshape(image.shape[0]//factor, factor, image.shape[1]//factor, factor, image.shape[2]), axis=(1,3))
        
        return image_new
    
    def plot_Mconf_nPhoton(self, Nimages):
        Mconf = np.zeros((3, 3))
        nPhoton = {}

        if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf_nPhoton_" + str(int(self.energy_tgt[1])) + "_" + str(int(self.energy_tgt[2])) + ".npz"):
            data = np.load(directory + "/results/image_reconstruction/" + self.identifier + "/Mconf_nPhoton_" + str(int(self.energy_tgt[1])) + "_" + str(int(self.energy_tgt[2])) + ".npz")
            Mconf = data['Mconf']
            nPhoton = data['nPhoton']
        
        # Plot Photons per Cluster Histogram
        fig, ax = plt.subplots(1, 1, figsize=[16,12], dpi=100)
        ax.hist(nPhoton, int(np.max(nPhoton) - np.min(nPhoton)))
        plt.savefig(directory + "/plots/" + self.identifier + "_nPhoton.png", dpi=100)
        plt.close()
        
        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=[30,30], dpi=300)
        ax.imshow(Mconf, cmap='gray', vmin=0, vmax=1, aspect=1)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        plt.savefig(directory + "/plots/" + self.identifier + "_Confusion_Matrix.png", dpi=100)
        plt.close()

    def plot_fowlkes_mallows_trend(self, Nphoton_rate_list, Nimages):
        Nrate = Nphoton_rate_list.size
    
        fowlkes_mallows_index = np.zeros(Nrate)
        fowlkes_mallows_index_range = np.zeros((Nrate, 2))

        for nph in range(Nrate):
            if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/fowlkes_mallows_pRate_" + str(Nphoton_rate_list[nph]) + ".npz"):
                data = np.load(directory + "/results/image_reconstruction/" + self.identifier + "/fowlkes_mallows_pRate_" + str(Nphoton_rate_list[nph]) + ".npz")
                fowlkes_mallows_index[nph] = np.nansum(data['fowlkes_mallows_index'])/(data['fowlkes_mallows_index'].size - np.sum(np.isnan(data['fowlkes_mallows_index'])))
                fowlkes_mallows_index_range[nph,0] = np.nanmin(data['fowlkes_mallows_index'])
                fowlkes_mallows_index_range[nph,1] = np.nanmax(data['fowlkes_mallows_index'])
            else:
                break
        
        fig, ax = plt.subplots(figsize=[16,12], dpi=100)
        lower_err = fowlkes_mallows_index_range[:,0]
        upper_err = fowlkes_mallows_index_range[:,1]
        mean_val = fowlkes_mallows_index.copy()
        errorbar = [mean_val-lower_err,upper_err-mean_val]
        ax.errorbar(Nphoton_rate_list, mean_val, yerr=errorbar, ecolor='lightskyblue', elinewidth=1.5, color='royalblue', marker='o', linewidth=2)
        plt.savefig(directory + "/plots/" + self.identifier + "_fowlkes_mallows_trend.png", dpi=100)
        plt.close()
        
        np.savez(directory + "/results/" + self.identifier + "_fowlkes_mallows_trend_data", fowlkes_mallows_index=fowlkes_mallows_index)

    def plot_phantom_images(self, Nphoton_rate_list):
        Nrate = Nphoton_rate_list.size
        
        if self.sampleID == 1:
            filename = 'AM'
        elif self.sampleID == 2:
            filename = 'AF'
        
        if self.sampleID == 1 or self.sampleID == 2:
            image_analytic, label_analytic = phantom.phantom(filename, self.dimSampleVoxel[2], self.indSampleVoxel, self.energy_list, self.energy_tgt_ind, self.xraySrc, self.IConc, self.GdConc,
                                                             self.G4directory)
            np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/image_analytic", image_analytic=image_analytic, label_analytic=label_analytic)

        for nph in range(Nrate):
            if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate_list[nph]) + ".npz"):
                with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate_list[nph]) + ".npz") as data:
                    image_true = data['image_true']
                    label_true = data['label_true']
                    image_reconstr = data['image_reconstr']
                    label_reconstr = data['label_reconstr']
            else:
                break
            
            if self.sampleID == 1 or self.sampleID == 2:
                fig, ax = plt.subplots(3, 4, figsize=[40,30], dpi=300)
                for i in range(self.energy_tgt.size-1):
                    vmin_analytic = np.min(image_analytic[:,:,i])
                    vmax_analytic = np.max(image_analytic[:,:,i])
                    ax[0,i].imshow(np.rot90(image_analytic[:,:,i]), interpolation='none', cmap='gray', vmin=vmin_analytic, vmax=vmax_analytic, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
                
                    vmin_true = np.min(image_true[:,:,i])
                    vmax_true = np.max(image_true[:,:,i])
                    ax[1,i].imshow(np.rot90(image_true[:,:,i]), interpolation='none', cmap='gray', vmin=vmin_true, vmax=vmax_true)
                    
                    vmin_reconstr = np.min(image_reconstr[:,:,i])
                    vmax_reconstr = np.max(image_reconstr[:,:,i])
                    ax[2,i].imshow(np.rot90(image_reconstr[:,:,i]), interpolation='none', cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
                
                vmax_analytic = np.max(label_analytic)
                ax[0,3].imshow(np.rot90(label_analytic[:,:,0]), interpolation='none', cmap='viridis', vmin=0, vmax=vmax_analytic, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
                
                vmax_true = np.max(label_true)
                ax[1,3].imshow(np.rot90(label_true), interpolation='none', cmap='viridis', vmin=0, vmax=vmax_true)
                
                vmax_reconstr = np.max(label_reconstr)
                ax[2,3].imshow(np.rot90(label_reconstr), interpolation='none', cmap='viridis', vmin=0, vmax=vmax_reconstr)
            
            elif self.sampleID == 3:
                fig, ax = plt.subplots(2, 4, figsize=[40,20], dpi=300)
                for i in range(self.energy_tgt.size-1):
                    vmin_true = np.min(image_true[:,:,i])
                    vmax_true = np.max(image_true[:,:,i])
                    ax[0,i].imshow(np.rot90(image_true[:,:,i]), interpolation='none', cmap='gray', vmin=vmin_true, vmax=vmax_true)
                    
                    vmin_reconstr = np.min(image_reconstr[:,:,i])
                    vmax_reconstr = np.max(image_reconstr[:,:,i])
                    ax[1,i].imshow(np.rot90(image_reconstr[:,:,i]), interpolation='none', cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
                
                vmax_true = np.max(label_true)
                ax[0,3].imshow(np.rot90(label_true), interpolation='none', cmap='viridis', vmin=0, vmax=vmax_true)
                
                vmax_reconstr = np.max(label_reconstr)
                ax[1,3].imshow(np.rot90(label_reconstr), interpolation='none', cmap='viridis', vmin=0, vmax=vmax_reconstr)

            plt.savefig(directory + "/plots/" + self.identifier + "_images10_Nrate" + str(Nphoton_rate_list[nph]) + ".png", dpi=300)
            plt.close()

@jit(nopython=True, cache=True)
def pca(xy_list):
    xy_mean = np.sum(xy_list, axis=0)/xy_list.shape[0]
    xy_std = np.zeros(xy_list.shape[1])
    for i in range(xy_list.shape[1]):
        xy_std[i] = xy_list[:,i].std()
    
    xy_norm = np.zeros((xy_list.shape[0], int(xy_list.shape[1]-np.sum(xy_std==0))))
    cnt = 0
    for i in range(xy_list.shape[1]):
        if xy_std[i] != 0:
            xy_norm[:,cnt] = (xy_list[:,i] - xy_mean[i])/xy_std[i]
            cnt += 1
    xy_cov = np.cov(xy_norm, rowvar=False, ddof=1)
    eigval, eigvec = np.linalg.eig(xy_cov)
    
    if eigval.size == 0:
        eigval = np.array([1.0])
        eigvec_out = np.array([1.0,1.0])
    else:
        eigvec_out = np.zeros(xy_list.shape[1])
        eigvec_out[xy_std!=0] = eigvec[:,np.argmax(eigval)]
        eigvec_out *= xy_std
    
    return np.max(eigval), eigvec_out

@jit(nopython=True, cache=True)
def weighted_kmeans(xy_list, weights, nCluster, maxiter, centroid):
    n_iter = 0
    centroid_prev = None
    
    while True:
        label = np.zeros(xy_list.shape[0])
        for xy in range(xy_list.shape[0]):
            distance = np.sqrt(np.sum((xy_list[xy,:][np.newaxis,:] - centroid)**2, axis=1))
            label[xy] = np.argmin(distance)
        
        if n_iter >= maxiter:
            break
        
        centroid_prev = centroid.copy()
        
        for nC in range(nCluster):
            centroid[nC,:] = np.sum(xy_list[label==nC,:]*weights[label==nC][:,np.newaxis], axis=0)/np.sum(weights[label==nC])
            if np.isnan(centroid[nC,:]).any():
                centroid[nC,:] = centroid_prev[nC,:]
        
        n_iter += 1
        if np.all(centroid - centroid_prev < 1e-2):
            break
    
    return centroid, label

@jit(nopython=True, cache=True)
def silhouette_coeff(nCluster, xy_list, weights, centroid, label):
    a = np.zeros(xy_list.shape[0])
    b = np.zeros(xy_list.shape[0])
    s = np.zeros(xy_list.shape[0])
    for n in range(nCluster):
        nCI = np.sum(label==n)
        if nCI == 1 or nCI == 0:
            s[label==n] = 0
        else:
            dSum = np.zeros(nCI)
            for i in range(nCI):
                d = np.zeros((nCI, 2))
                d[:,0] = xy_list[label==n,0] - xy_list[label==n,0][i]
                d[:,1] = xy_list[label==n,1] - xy_list[label==n,1][i]
                dSum += np.sqrt(np.sum(d**2, axis=1))
            a[label==n] = (1/(nCI - 1))*dSum
            
            for m in range(nCluster):
                if m != n:
                    nCJ = np.sum(label==m)
                    dSum = np.zeros(nCI)
                    for i in range(nCJ):
                        d = np.zeros((nCI, 2))
                        d[:,0] = xy_list[label==n,0] - xy_list[label==m,0][i]
                        d[:,1] = xy_list[label==n,1] - xy_list[label==m,1][i]
                        dSum += np.sqrt(np.sum(d**2, axis=1))
                    if nCJ > 0:
                        if np.sum(b[label==n]) == 0:
                            b[label==n] = dSum/nCJ
                        else:
                            b[label==n] = np.minimum(b[label==n], dSum/nCJ)
            
            s[label==n] = (b[label==n] - a[label==n])/np.maximum(a[label==n], b[label==n])
    
    sc = np.sum(s)/s.size
    
    return sc

@jit(nopython=True, cache=True)
def fowlkes_mallows_index(label_true, label_reconstr):
    TP = 0
    FP = 0
    FN = 0
    
    for i in range(label_true.size):
        pair_true = label_true == label_true[i]
        pair_reconstr = label_reconstr == label_reconstr[i]
    
        TP += np.sum(pair_true*pair_reconstr)
        FP += np.sum(~pair_true*pair_reconstr)
        FN += np.sum(pair_true*~pair_reconstr)
    
    TP -= label_true.size
    
#        for i in range(label_true.size):
#            for j in range(label_true.size):
#                if i != j:
#                    pair_true = label_true[i] == label_true[j]
#                    pair_reconstr = label_reconstr[i] == label_reconstr[j]
#                    
#                    if pair_true and pair_reconstr:
#                        TP += 1
#                    elif not pair_true and pair_reconstr:
#                        FP += 1
#                    elif pair_true and not pair_reconstr:
#                        FN += 1
    
    return np.sqrt((TP/(TP + FP))*(TP/(TP + FN)))

def label_accuracy(label_true, label_reconstr):
    nlabels = int(np.max((np.max(label_true), np.max(label_reconstr)))) + 1
    label_reconstr = label_reconstr[label_true!=0]
    label_true = label_true[label_true!=0]
    
    confusion_matrix = np.zeros((nlabels, nlabels), dtype=int)
    for i in range(nlabels):
        for j in range(nlabels):
            confusion_matrix[i,j] = np.sum((label_true == i) & (label_reconstr == j))
    
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    optimal_matches = confusion_matrix[row_ind,col_ind].sum()
    total_pixels = label_true.size
    
    accuracy = optimal_matches/total_pixels
    
    return accuracy

@jit(nopython=True, cache=True)
def dbscan(xy_list, noise_threshold=0.001, dist_neighbor=0.1):
    n_pts = xy_list.shape[0]

    label = np.zeros(n_pts)
    
    cluster_index = 0
    visited = np.zeros(n_pts)
    
    for nxy in range(n_pts):
        if visited[nxy]:
            continue
        visited[nxy] = True
        
        # Find Neighboring Points
        dist = np.sqrt(np.sum((xy_list[nxy,:][np.newaxis,:] - xy_list)**2, axis=1))
        neighbors = np.where(dist < dist_neighbor)[0]
        
        if neighbors.size < noise_threshold*n_pts:
            # if under threshold, the point is labeled as noise
            label[nxy] = 0
        else:
            # if over threshold, the point is a core point of a cluster
            cluster_index += 1
            label[nxy] = cluster_index
            
            cnt = 0
            while cnt < neighbors.size:
                neighbor_index = neighbors[cnt]
                if not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    dist = np.sqrt(np.sum((xy_list[neighbor_index,:][np.newaxis,:] - xy_list)**2, axis=1))
                    # identify 2nd neighbors
                    second_neighbors = np.where(dist < dist_neighbor)[0]
                    if second_neighbors.size >= noise_threshold*n_pts:
                        # if the neighbor point is a core point, add all 2nd neighbors to the search list
                        neighbors = np.append(neighbors, second_neighbors)
                
                if label[neighbor_index] == 0:
                    # any point in the neighborhood of a core point belongs to the same cluster
                    label[neighbor_index] = cluster_index
                
                cnt += 1
    
    return label

def image_normalization_cost(x0, img_ref, img):
    scale = x0[0]
    minVal = x0[1]
    
    img_norm = scale*img + minVal
    
    cost = np.sqrt(np.sum((img_ref - img_norm)**2))
    
    return cost

if __name__ == '__main__':
    if rank == 0:
        verbose = True
    else:
        verbose = False

    RGB = True
    if RGB:
        RGB_scintillator = True
        Nlayers = 3
        scintillator_material = np.array([2,7,6])
        scintillator_thickness = np.array([0.3,0.3,0.3]) # 0.775962,0.984305,0.915477
        distribution_datafile = 'ZnSe_Gadox_NaI_67keV_300um'
        identifier = 'ZnSe_Gadox_NaI_67keV_300um'
        rank_start = 0
    else:
        RGB_scintillator = False
        Nlayers = 1
        scintillator_material = np.array([2])
        scintillator_thickness = np.array([0.1+0.1+0.1])
        distribution_datafile = 'ZnSe_67keV_300um_test'
        identifier = 'ZnSe_67keV_300um_test'
        rank_start = 0
    
    if comm.rank == 0:
        if not os.path.exists(directory + "/data/reconstruction_examples/" + identifier):
            os.mkdir(directory + "/data/reconstruction_examples/" + identifier)
        
        if not os.path.exists(directory + "/results/image_reconstruction/" + identifier):
            os.mkdir(directory + "/results/image_reconstruction/" + identifier)

    #G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main"
    #G4directory = "/home/minseokhwan/geant4/G4_Nanophotonic_Scintillator-main"
    simulation = geant4(G4directory = "/home/minseokhwan/xray_projects/geant4/G4_Nanophotonic_Scintillator-main",
                        RGB_scintillator = RGB_scintillator, #---------------------------------------------------------- Change
                        sampleID = 0, # 0: None, 1:AM, 2:AF, 3:Cylindrical #----------------------------------------------- Change
                        dimSampleVoxel = np.array([2.137,8,0.0])/np.array([400,400,1]), # in cm
                        indSampleVoxel = np.array([[0,253], # 0...253 indexing
                                                   [119,189], # 0...222 indexing
                                                   [63,64]]), # 1...128 indexing
                        IConc = 0.01,
                        GdConc = 0.01,
                        dimScint = np.array([1.5,1.5]), # in cm
                        dimDetVoxel = np.array([10,10,0.1]), # in um
                        NDetVoxel = np.array([1280,1280,1]),
                        gapSampleScint = 0.0, # in cm
                        gapScintDet = 0.001, # in cm
                        Nlayers = Nlayers, #---------------------------------------------------------------------- Change
                        scintillator_material = scintillator_material, # 1:YAGCe, 2:ZnSeTe, 3:LYSOCe, 4:CsITl, 5:GSOCe, 6:NaITl, 7:GadoxTb, 8:YAG:Yb #- Change
                        scintillator_thickness = scintillator_thickness, # in mm #---------- Change 0.222604,0.313417,0.337213 | 0.076081,0.011322,0.073472
                        check_overlap = 0,
                        energy_range = np.array([16,67]), # in keV
                        energy_tgt = np.array([16,33,50,67]), # np.linspace(10, 150, 6), # in keV
                        energy_tgt_index_imed = np.array([0,1]),
                        energy_bins = 52,
                        d_mean_bins = 100,
                        theta_filt = 90*np.pi/180, #15*np.pi/180 / None
                        xray_target_material = 'W',
                        xray_operating_V = 67, # in keV
                        xray_tilt_theta = 12, # in deg
                        xray_filter_material = ['Al'],
                        xray_filter_thickness = [0.3], # in cm
                        verbose = verbose,
                        distribution_datafile = distribution_datafile,  #---------------------------------------------------- Change
                        identifier = identifier, #-------------------------------------------------------- Change RGB_150keV_nogap_phantom
                        rank_start = rank_start, #----------------------------------------------------------------- Change
                        )
    
#    if rank == 0:
#        simulation.cmake()
#        simulation.make()
#    else:
#        time.sleep(120)
    
    # Generate Energy vs. Radius Distribution Data
    if verbose:
        print('\n### Collecting Energy Distribution Data', flush=True)
        print('    ', end='', flush=True)
    for i in range(100): # originally 100
        if verbose:
            if i%10 == 0:
                print(i, end='', flush=True)
            else:
                print('/', end='', flush=True)
        simulation.generate_energy_distribution_data(Nruns_per_energy=1000) # originally 10000
    if verbose:
        print(i+1, flush=True)
    
    # Evaluate Energy Reconstruction
#    E_list1 = np.linspace(20, 130, 12)
#    E_list2 = np.linspace(30, 140, 12)
#    E_list1 = np.array([33])
#    E_list2 = np.array([50])    
#    accuracy = np.zeros((E_list1.size, E_list2.size))
#    simulation.xraySrc = np.ones(simulation.energy_bins)/simulation.energy_bins
#    for i in range(E_list1.size):
#        for j in range(E_list2.size):
#            if E_list1[i] < E_list2[j]:
#                Nphoton = int(1e6)
#                t1 = time.time()
#                Nimages = int(Nphoton)
#                simulation.energy_tgt[1] = E_list1[i]
#                simulation.energy_tgt[2] = E_list2[j]
#                accuracy[i,j] = simulation.test_energy_reconstruction(Nimages)
#                if comm.rank == 0:
#                    simulation.plot_Mconf_nPhoton(Nimages)
#                comm.Barrier()
#                t2 = time.time()
#                if verbose:
#                    print('    | Time Taken: ' + str(t2-t1) + ' s')
#                if rank == 0:
#                    np.savez(directory + "/results/image_reconstruction/" + identifier + "/energy_accuracy", accuracy=accuracy)
    
    # Evaluate Spatial Reconstruction
#    Nphoton_rate_list = np.array([10,100,1000]).astype(int)#np.hstack((np.linspace(10, 100, 10), np.linspace(200, 1000, 9))).astype(int)
#    Nphoton_rate_list_plot = np.array([10,100,1000]).astype(int)#np.hstack((np.linspace(10, 100, 10), np.linspace(200, 1000, 9))).astype(int)
#    Nphoton = 1000#1e5
#    for Nphoton_rate in Nphoton_rate_list:
#        t1 = time.time()
#        Nimages = int(Nphoton/Nphoton_rate)
#        simulation.test_spatial_reconstruction(Nphoton_rate, Nimages)
#        if comm.rank == 0:
#            simulation.plot_fowlkes_mallows_trend(Nphoton_rate_list_plot, Nimages)
#        comm.Barrier()
#        t2 = time.time()
#        if verbose:
#            print('    | Time Taken: ' + str(t2-t1) + ' s')

    # Evaluate Signal-to-Noise Ratio
#    Nphoton_max = 1e9
#    Nphoton_rate = int(1e2)
#    Nimages_list = (np.linspace(1e-1, 1, 10)*Nphoton_max/Nphoton_rate).astype(int)
#    Nimages_list_plot = (np.linspace(1e-1, 1, 10)*Nphoton_max/Nphoton_rate).astype(int)
#    SNR = np.zeros(0)
#    for Nimages in Nimages_list:
#        t1 = time.time()
#        simulation.image_true *= 0
#        simulation.image_reconstr *= 0
#        SNR_temp = simulation.compute_SNR(Nphoton_rate, Nimages)
#        SNR = np.append(SNR, SNR_temp)
#        if comm.rank == 0:
#            np.savez(directory + "/results/image_reconstruction/" + simulation.identifier + "/SNR_all", SNR=SNR)
#        comm.Barrier()
#        t2 = time.time()
#        if verbose:
#            print('    | Time Taken: ' + str(t2-t1) + ' s')
    
    # Simulate Phantom Image
#    Nphoton_rate_list = np.linspace(900, 900, 1).astype(int)
#    Nphoton_rate_list_plot = np.linspace(900, 900, 1).astype(int)
#    Nphoton_total = int(1e8)
#    for Nphoton_rate in Nphoton_rate_list:
#        Nimages = int(Nphoton_total/Nphoton_rate)
#        simulation.image_true *= 0
#        simulation.image_reconstr *= 0
#        if comm.rank == 0:
#            with np.load(directory + "/results/image_reconstruction/" + simulation.identifier + "/arrays_pRate" + str(Nphoton_rate) + ".npz") as data: # 186e6 done
#                simulation.image_true = data['image_true']
#                simulation.image_reconstr = data['image_reconstr']
#        for i in range(9, 100):
#            t1 = time.time()
#            if verbose:
#                print('\n### Run ' + str(i+1) + ' of 100', flush=True)
#            if comm.rank == 0:
#                simulation.plot_phantom_images(Nphoton_rate_list_plot)
#            simulation.test_phantom(Nphoton_rate, Nimages)
#            if comm.rank == 0:
#                simulation.plot_phantom_images(Nphoton_rate_list_plot)
#            comm.Barrier()
#            t2 = time.time()
#            if verbose:
#                print('    | Time Taken: ' + str(t2-t1) + ' s', flush=True)

    # Simulate Simple Model Phantom
    # simulation.test_phantom_simple_model(Npixel_side=150, system='medical') #150, 225, 375
    # simulation.test_phantom_CT(ind_slice=179, theta=np.linspace(0, 180, 361))