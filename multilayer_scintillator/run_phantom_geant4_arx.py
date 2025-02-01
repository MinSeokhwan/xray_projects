import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numba import jit
import autograd.numpy as npa
from scipy.optimize import least_squares, Bounds, minimize
from scipy.cluster.vq import kmeans2
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, anderson
import diptest
import matplotlib.pyplot as plt
from itertools import product
import subprocess
import uproot
import time
import util.read_txt as txt
import multilayer_scintillator.xray_spectrum_generator as xray
import multilayer_scintillator.generate_linAttCoeff_dat as phantom

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
                 xray_target_material=None,
                 xray_operating_V=None,
                 xray_tilt_theta=None,
                 xray_filter_material=None,
                 xray_filter_thickness=None,
                 verbose=False,
                 distribution_datafile = '',
                 identifier='',
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
        self.dzWorld = 1.01*(0.1 + self.dimDet[2]/1e4 + self.dimSample[2] + gapSampleScint + np.sum(self.scintillator_thickness)/10 + gapScintDet + self.dimDet[2]/1e4) # cm
        
        self.check_overlap = check_overlap
        
        self.energy_list = np.linspace(energy_range[0], energy_range[1], energy_bins)
        self.energy_bins = energy_bins
        self.energy_tgt = energy_tgt
        self.energy_tgt_index_imed = energy_tgt_index_imed
        self.energy_tgt_ind = np.zeros(energy_tgt.size).astype(int)
        for i_tgt in range(energy_tgt.size):
            self.energy_tgt_ind[i_tgt] = np.argmin(np.abs(self.energy_list - self.energy_tgt[i_tgt]))
        
        # proc = subprocess.Popen([".", "/home/gridsan/smin/geant4/geant4-v11.1.2-install/share/Geant4/geant4make/geant4make.sh"])
        
        self.sensRGB = txt.read_txt(directory + "/data/rgb_sensitivity")
        self.RGB_basis_done = False
        # self.set_RGB_basis()
        
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
        
        self.verbose = verbose
        self.distribution_datafile = distribution_datafile
        self.identifier = identifier
        
        self.d_mean_list = np.linspace(0, np.max(self.dimDet[:2])/2e4, d_mean_bins) # mm
        self.d_mean_bins = d_mean_bins
        
        if os.path.exists(directory + "/data/energy_distribution_data_" + distribution_datafile + ".npz"):
            self.energy_distribution = np.load(directory + "/data/energy_distribution_data_" + distribution_datafile + ".npz")['energy_distribution']
            #self.radius_vs_Nphoton = np.load(directory + "/data/energy_distribution_data_" + distribution_datafile + ".npz")['radius_vs_Nphoton']
        else:
            self.energy_distribution = None
            #self.radius_vs_Nphoton = None
    
    def cmake(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["cmake", ".."])
        time.sleep(10)
        
    def make(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["make", "-j4"])
        time.sleep(20)
    
    def write_dat_file(self, name, data):
        with open(self.G4directory + "/build/" + name + ".dat", "w") as dat:
            for nd in range(data.size-1):
                dat.write(data[nd] + "\n")
            dat.write(data[-1])
    
    def make_simulation_macro(self, Nruns, source_energy=None):
        with open(self.G4directory + "/build/run_detector_rank" + str(comm.rank) + ".mac", "w") as mac:
            mac.write("/system/root_file_name output" + str(comm.rank) + ".root\n")
        
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
            # mac.write("/control/execute vis.mac\n\n")
            
            mac.write("/gps/particle gamma\n")
            mac.write("/gps/direction 0 0 1\n")
            if source_energy is None:
                mac.write("/gps/pos/type Plane\n")
                mac.write("/gps/pos/shape Square\n")
                mac.write("/gps/pos/centre 0 0 " + str(np.round(-self.dzWorld/2.02, 6)) + " cm\n")
                mac.write("/gps/pos/halfx " + str(np.round(self.dimDet[0]/2e4, 6)) + " cm\n")
                mac.write("/gps/pos/halfy " + str(np.round(self.dimDet[1]/2e4, 6)) + " cm\n")
                
                mac.write("/gps/ene/type User\n")
                mac.write("/gps/ene/emspec 0\n")
                mac.write("/gps/hist/type energy\n")
                mac.write("/gps/hist/point 0. 0.\n")
                for ne in range(self.energy_bins):
                    mac.write("/gps/hist/point " + str(np.round(self.energy_list[ne]/1e3, 6)) + " " + str(np.round(self.xraySrc[ne], 6)) + "\n")
            else:
                mac.write("/gps/pos/type Point\n")
                mac.write("/gps/pos/centre 0 0 " + str(np.round(-self.dzWorld/2.02, 6)) + " cm\n")
                
                mac.write("/gps/ene/type Mono\n")
                mac.write("/gps/ene/mono " + str(np.round(source_energy, 6)) + " keV\n")
            mac.write("/run/beamOn " + str(int(Nruns)))
            
#            mac.write("/gun/momentumAmp " + str(np.round(source_energy, 6)) + " keV\n")
#            mac.write("/gun/position " + str(np.round(source_xy[0], 6)) + " " + str(np.round(source_xy[1], 6)) + " " + str(np.round(-self.dzWorld/2.02, 6)) + " cm\n")
#            mac.write("/run/beamOn " + str(int(Nruns)))
        
    def read_output(self, Nruns):
        if os.path.exists(self.G4directory + "/build/output" + str(comm.rank) + ".root"):
            os.remove(self.G4directory + "/build/output" + str(comm.rank) + ".root")
    
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["./NS", "run_detector_rank" + str(comm.rank) + ".mac"])
        
        while True:
            if os.path.exists(self.G4directory + "/build/output" + str(comm.rank) + ".root"):
                break
            time.sleep(1)
        
        while True:
            try:
                root_file = uproot.open(self.G4directory + "/build/output" + str(comm.rank) + ".root")
                photons = root_file['Photons']
                hits = root_file['Hits']
                source = root_file['Source']
                break
            except:
                time.sleep(1)
        
        creatorProcess = [t for t in photons["fCreatorProcess"].array()] # run 10000 or more at a time and use fEvent to determine which is which
        scint_photon_ind = np.array([t == 'Scintillation' for t in creatorProcess])
        wvl_list = np.array(photons["fWlen"].array())[scint_photon_ind]
        
        eventHits = np.array(hits["fEvent"].array())[scint_photon_ind]
        x_list = np.array(hits["fX"].array())[scint_photon_ind]
        y_list = np.array(hits["fY"].array())[scint_photon_ind]
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
            
#        np.savez(directory + "/data/debug_read_output" + str(comm.rank), wvl_list=wvl_list, eventHits=eventHits, xy_list=xy_list, energySrc_list=energySrc_list, xySrc_list=xySrc_list,
#                 creatorProcess=np.array(photons["fCreatorProcess"].array()), Wlen=np.array(photons["fWlen"].array()), fEventPhoton=np.array(photons["fEvent"].array()),
#                 fEventHits=np.array(hits["fEvent"].array()), fEventSrc=np.array(source["fEvent"].array()))
        
        return wvl_dict, xy_dict, energySrc_list, xySrc_list
    
    def set_RGB_basis(self):
        scintillator_names = ['YAGCe','ZnSeTe','LYSOCe','CsITl','GSOCe']
        self.RGB_basis = np.zeros((3, self.scintillator_material.size))
        for nmat in range(self.scintillator_material.size):
            emission = txt.read_txt(directory + "/data/emission" + scintillator_names[self.scintillator_material[nmat]-1])
            self.RGB_basis[:,nmat] = self.RGB_compute(emission[:,1], emission[:,0])
        self.RGB_basis_done = True
    
    def RGB_compute(self, emission, wavelength):
        sensRGB_interp = np.zeros((wavelength.size, 3))
        for i in range(3):
            sensRGB_interp[:,i] = np.interp(wavelength, self.sensRGB[:,0], self.sensRGB[:,i+1])
            sensRGB_interp[:,i] /= 100
            
        RGB = np.zeros(3)
        for i in range(3):
            RGB[i] = np.trapz(sensRGB_interp[:,i]*emission, wavelength)
        RGB_original = RGB.copy()
        
        if self.RGB_basis_done:
            RGB = np.linalg.inv(self.RGB_basis) @ RGB
            
        RGB /= np.max(RGB)
        
        return RGB
    
    def RGB_readings(self, wvl_list):
        RGB = np.zeros((wvl_list.size, 3))
        for i in range(3):
            RGB[:,i] = np.interp(wvl_list, self.sensRGB[:,0], self.sensRGB[:,i+1])
            RGB[:,i] /= 100
        
        if self.RGB_basis_done:
            RGB = (np.linalg.inv(self.RGB_basis) @ RGB.T).T
        
        return RGB
    
    def RGB_readings2(self, wvl_list):
        RGB = np.zeros(wvl_list.size)
        for i in range(wvl_list.size):
            RGB[:,i] = np.interp(wvl_list, self.sensRGB[:,0], self.sensRGB[:,i+1])
            RGB[:,i] /= 100
        
        if self.RGB_basis_done:
            RGB = (np.linalg.inv(self.RGB_basis) @ RGB.T).T
        
        return RGB
    
    def data_interpretation(self, xy_list):
#        if self.RGB_scintillator:
#            if xy_list.size == 2:
#                centroidR = xy_list.copy()
#                centroidG = xy_list.copy()
#                centroidB = xy_list.copy()
#                labelR = np.zeros(1)
#                labelG = np.zeros(1)
#                labelB = np.zeros(1)
#                
#            elif xy_list.size == 4:
#                centroidR = (RGB_list[1,0]/np.sum(RGB_list[:,0])*(xy_list[1,:] - xy_list[0,:]) + xy_list[0,:]).reshape(1, -1)
#                centroidG = (RGB_list[1,1]/np.sum(RGB_list[:,1])*(xy_list[1,:] - xy_list[0,:]) + xy_list[0,:]).reshape(1, -1)
#                centroidB = (RGB_list[1,2]/np.sum(RGB_list[:,2])*(xy_list[1,:] - xy_list[0,:]) + xy_list[0,:]).reshape(1, -1)
#                labelR = np.zeros(2)
#                labelG = np.zeros(2)
#                labelB = np.zeros(2)
#                
#            else:
#                centroidR, labelR = self.weighted_kmeans(xy_list, RGB_list[:,0], nCluster=1, maxiter=100)
#                centroidG, labelG = self.weighted_kmeans(xy_list, RGB_list[:,1], nCluster=1, maxiter=100)
#                centroidB, labelB = self.weighted_kmeans(xy_list, RGB_list[:,2], nCluster=1, maxiter=100)
#            
#            d_meanR = np.sum(np.sqrt(np.sum((centroidR[0,:][np.newaxis,:] - xy_list)**2, axis=1))*RGB_list[:,0])/np.sum(RGB_list[:,0])
#            d_meanG = np.sum(np.sqrt(np.sum((centroidG[0,:][np.newaxis,:] - xy_list)**2, axis=1))*RGB_list[:,1])/np.sum(RGB_list[:,1])
#            d_meanB = np.sum(np.sqrt(np.sum((centroidB[0,:][np.newaxis,:] - xy_list)**2, axis=1))*RGB_list[:,2])/np.sum(RGB_list[:,2])
#            ind_d_meanR = np.argmin(np.abs(d_meanR - self.d_mean_list))
#            ind_d_meanG = np.argmin(np.abs(d_meanG - self.d_mean_list))
#            ind_d_meanB = np.argmin(np.abs(d_meanB - self.d_mean_list))
#            
#            if comm.rank == 0:
#                np.savez(directory + "/data/debug_data_interp_RGB_" + str(int(xy_list.size/2)), xy_list=xy_list, RGB_list=RGB_list,
#                         centroidR=centroidR, centroidG=centroidG, centroidB=centroidB,
#                         labelR=labelR, labelG=labelG, labelB=labelB,
#                         d_meanR=d_meanR, d_meanG=d_meanG, d_meanB=d_meanB,
#                         ind_d_meanR=ind_d_meanR, ind_d_meanG=ind_d_meanG, ind_d_meanB=ind_d_meanB, d_mean_list=self.d_mean_list
#            
#            return ind_d_meanR, ind_d_meanG, ind_d_meanB)
#            
#        else:
        if xy_list.size == 2:
            centroid = xy_list.copy()
            label = np.zeros(1)
            
        elif xy_list.size == 4:
            centroid = ((xy_list[1,:] - xy_list[0,:])/2 + xy_list[0,:]).reshape(1, -1)
            label = np.zeros(2)
            
        else:
            centroid, label = weighted_kmeans(xy_list, np.ones(xy_list.shape[0]), nCluster=1, maxiter=20, centroid=(self.dimDet[:2]/1e3).reshape(1,-1)) # weights-->RGB_list
        
        npts = xy_list.shape[0]
        d_mean = np.sum(np.sqrt(np.sum((centroid[0,:][np.newaxis,:] - xy_list)**2, axis=1)))/npts
        ind_d_mean = np.argmin(np.abs(d_mean - self.d_mean_list))
        
#        if comm.rank == 0:
#            np.savez(directory + "/data/debug_data_interp_" + str(int(xy_list.size/2)), xy_list=xy_list,
#                     centroid=centroid, label=label, d_mean=d_mean, ind_d_mean=ind_d_mean, d_mean_list=self.d_mean_list)
    
        return ind_d_mean, d_mean
    
    def reconstruct_image(self, source_energy_list, source_xy_list, xy_list, RGB_list, event_label, nCluster=None, nTrials=10):
        image_reconstr_temp = np.zeros_like(self.image_reconstr).astype(np.float32)
    
        if self.RGB_scintillator:
            nk = nCluster
            
            xy_list_raw = xy_list.copy()
            RGB_list_raw = RGB_list.copy()
            xy_list = xy_list_raw[np.sum(RGB_list_raw<0.01, axis=1)==1,:]
            RGB_list = RGB_list_raw[np.sum(RGB_list_raw>0.1, axis=1)==1,:]
            event_label_filt = event_label[np.sum(RGB_list_raw>0.1, axis=1)==1]
            
            nPhotonR = np.zeros(0)
            nPhotonG = np.zeros(0)
            nPhotonB = np.zeros(0)
            source_ind_RGB = np.zeros(nCluster).astype(int)
            for i in range(nCluster):
                if np.sum(event_label_filt==i) == 0:
                    source_ind_RGB[i] = -1
                else:
                    source_ind_RGB[i] = np.argmax(np.sum(RGB_list[event_label_filt==i,:], axis=0))
                    if source_ind_RGB[i] == 0:
                        nPhotonR = np.append(nPhotonR, np.sum(event_label_filt==i))
                    elif source_ind_RGB[i] == 1:
                        nPhotonG = np.append(nPhotonG, np.sum(event_label_filt==i))
                    elif source_ind_RGB[i] == 2:
                        nPhotonB = np.append(nPhotonB, np.sum(event_label_filt==i))
            
            min_cluster_size = np.array([20,120,20]) # 100,200,200
            max_cluster_size = np.array([450,1200,300]) # 300,600,600
            xy_list2 = None
            RGB_list2 = None
            event_label_filt2 = None
            for i in range(3):
                if np.sum(np.argmax(RGB_list, axis=1)==i) > min_cluster_size[i]:
                    xy_i = xy_list[np.argmax(RGB_list, axis=1)==i,:]
                    RGB_i = RGB_list[np.argmax(RGB_list, axis=1)==i,:]
                    event_label_i = event_label_filt[np.argmax(RGB_list, axis=1)==i]
                
                    if xy_list2 is None:
                        xy_list2 = xy_i.copy()
                        RGB_list2 = RGB_i.copy()
                        event_label_filt2 = event_label_i.copy()
                    else:
                        xy_list2 = np.vstack((xy_list2, xy_i))
                        RGB_list2 = np.vstack((RGB_list2, RGB_i))
                        event_label_filt2 = np.hstack((event_label_filt2, event_label_i))
            
            if RGB_list2 is not None:
                sc_ground_truth = silhouette_coeff(int(np.max(event_label_filt)), xy_list, np.ones(xy_list.shape[0]), centroid=source_xy_list, label=event_label_filt)
            
                energy_distribution_temp = self.energy_distribution.copy()
                energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=(0,1))[np.newaxis,np.newaxis,:]
                energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=2)[:,:,np.newaxis]
                energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_distribution.shape[1], self.energy_tgt.size-1))
                for i_tgt in range(self.energy_tgt.size-1):
                    energy_distribution_norm[:,:,i_tgt] = np.sum(energy_distribution_temp[:,:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=2)
                
                centroid0 = None
                label0 = np.zeros(xy_list2.shape[0])
                ind_RGB = None
                sc_RGB = np.zeros(3)
                xrayInc_all = None
                d_mean_all = np.zeros(0)
                for i in range(3):
                    mask_i = np.argmax(RGB_list2, axis=1)==i
                    xy_i = xy_list2[mask_i,:]
                    RGB_i = RGB_list2[mask_i,:]
                
                    if xy_i.shape[0] > min_cluster_size[i]:
                        centroid, label = self.gmeans(xy_i, 20, 20, min_cluster_size=min_cluster_size[i], max_cluster_size=max_cluster_size[i])
                        if centroid0 is None:
                            centroid0 = centroid.reshape(-1, 2)
                            label0[mask_i] = label.copy()
                            ind_RGB = i*np.ones(centroid.shape[0])
                        else:
                            centroid0 = np.vstack((centroid0, centroid))
                            label0[mask_i] = label + np.max(label0) + 1
                            ind_RGB = np.hstack((ind_RGB, i*np.ones(centroid.shape[0])))
                        nk_i = centroid.shape[0]
                        
                        if centroid.shape[0] > 1:
                            sc_RGB[i] = silhouette_coeff(nk_i, xy_i, np.ones(xy_i.shape[0]), centroid=centroid, label=label)
        
                        d_mean = np.zeros(nk_i)
                        
                        for k in range(nk_i):
                            npts = int(np.sum(label==k))
                            if npts > 0:
                                xy_i_temp = xy_i[label==k,:]
                                
                                d_mean[k] = np.sum(np.sqrt(np.sum((np.tile(centroid[k,:], (npts,1)) - xy_i_temp)**2, axis=1)))/npts
                            
                                ind_d_mean = np.argmin(np.abs(d_mean[k] - self.d_mean_list))
                                
                                while True:
                                    xrayInc = energy_distribution_norm[i,ind_d_mean,:]
                                    # xrayInc = savgol_filter(xrayInc, 10, 5)
                                    # xrayInc[xrayInc<0] = 0
                                    if np.sum(xrayInc) > 0:
                                        # xrayInc /= np.sum(xrayInc)
                                        if xrayInc_all is None:
                                            xrayInc_all = xrayInc.copy()
                                        else:
                                            xrayInc_all = np.vstack((xrayInc_all, xrayInc))
                                        break
                                    ind_d_mean -= 1
                            
                                ind_x0 = int((centroid[k,0] - self.dimDetVoxel[0]/2e3)//(self.dimDetVoxel[0]/1e3))
                                x0 = self.dimDetVoxel[0]/2e3 + ind_x0*self.dimDetVoxel[0]/1e3
                                ind_x1 = ind_x0 + 1
                                x1 = self.dimDetVoxel[0]/2e3 + ind_x1*self.dimDetVoxel[0]/1e3
                                    
                                ind_y0 = int((centroid[k,1] - self.dimDetVoxel[1]/2e3)//(self.dimDetVoxel[1]/1e3))
                                y0 = self.dimDetVoxel[1]/2e3 + ind_y0*self.dimDetVoxel[1]/1e3
                                ind_y1 = ind_y0 + 1
                                y1 = self.dimDetVoxel[1]/2e3 + ind_y1*self.dimDetVoxel[1]/1e3
                    
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
                    
                                image_reconstr_temp[ind_x0,ind_y0,:] += xrayInc*ratio[0]
                                image_reconstr_temp[ind_x0,ind_y1,:] += xrayInc*ratio[1]
                                image_reconstr_temp[ind_x1,ind_y0,:] += xrayInc*ratio[2]
                                image_reconstr_temp[ind_x1,ind_y1,:] += xrayInc*ratio[3]
                    
                                self.image_reconstr[ind_x0,ind_y0,:] += xrayInc*ratio[0]
                                self.image_reconstr[ind_x0,ind_y1,:] += xrayInc*ratio[1]
                                self.image_reconstr[ind_x1,ind_y0,:] += xrayInc*ratio[2]
                                self.image_reconstr[ind_x1,ind_y1,:] += xrayInc*ratio[3]
                        
                        d_mean_all = np.hstack((d_mean_all, d_mean))
            
                nk = centroid0.shape[0]
                sc = np.mean(sc_RGB)
                fm = fowlkes_mallows_index(event_label_filt2, label0)
                
                if comm.rank == 0:
                    if os.path.exists(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min.npz"):
                        data = np.load(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min.npz")
                        if fm < data['fm']:
                            np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min", d_mean_all=d_mean_all, xrayInc=xrayInc_all,
                                     ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, xy_list=xy_list, xy_list2=xy_list2, RGB_list=RGB_list,
                                     RGB_list2=RGB_list2, source_energy_list=source_energy_list, source_xy_list=source_xy_list, centroid0=centroid0, label0=label0, event_label=event_label,
                                     sc=sc, ind_RGB=ind_RGB, xy_list_raw=xy_list_raw, RGB_list_raw=RGB_list_raw, sc_RGB=sc_RGB, fm=fm, energy_distribution_norm=energy_distribution_norm,
                                     event_label_filt=event_label_filt, event_label_filt2=event_label_filt2, sc_ground_truth=sc_ground_truth, centroid=centroid, label=label, xy_i=xy_i)
                    else:
                        np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min", d_mean_all=d_mean_all, xrayInc=xrayInc_all,
                                 ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, xy_list=xy_list, xy_list2=xy_list2, RGB_list=RGB_list,
                                 RGB_list2=RGB_list2, source_energy_list=source_energy_list, source_xy_list=source_xy_list, centroid0=centroid0, label0=label0, event_label=event_label,
                                 sc=sc, ind_RGB=ind_RGB, xy_list_raw=xy_list_raw, RGB_list_raw=RGB_list_raw, sc_RGB=sc_RGB, fm=fm, energy_distribution_norm=energy_distribution_norm,
                                 event_label_filt=event_label_filt, event_label_filt2=event_label_filt2, sc_ground_truth=sc_ground_truth, centroid=centroid, label=label, xy_i=xy_i)
                    
                    if os.path.exists(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max.npz"):
                        data = np.load(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max.npz")
                        if fm > data['fm']:
                            np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max", d_mean_all=d_mean_all, xrayInc=xrayInc_all,
                                     ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, xy_list=xy_list, xy_list2=xy_list2, RGB_list=RGB_list,
                                     RGB_list2=RGB_list2, source_energy_list=source_energy_list, source_xy_list=source_xy_list, centroid0=centroid0, label0=label0, event_label=event_label,
                                     sc=sc, ind_RGB=ind_RGB, xy_list_raw=xy_list_raw, RGB_list_raw=RGB_list_raw, sc_RGB=sc_RGB, fm=fm, energy_distribution_norm=energy_distribution_norm,
                                     event_label_filt=event_label_filt, event_label_filt2=event_label_filt2, sc_ground_truth=sc_ground_truth, centroid=centroid, label=label, xy_i=xy_i)
                    else:
                        np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max", d_mean_all=d_mean_all, xrayInc=xrayInc_all,
                                 ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, xy_list=xy_list, xy_list2=xy_list2, RGB_list=RGB_list,
                                 RGB_list2=RGB_list2, source_energy_list=source_energy_list, source_xy_list=source_xy_list, centroid0=centroid0, label0=label0, event_label=event_label,
                                 sc=sc, ind_RGB=ind_RGB, xy_list_raw=xy_list_raw, RGB_list_raw=RGB_list_raw, sc_RGB=sc_RGB, fm=fm, energy_distribution_norm=energy_distribution_norm,
                                 event_label_filt=event_label_filt, event_label_filt2=event_label_filt2, sc_ground_truth=sc_ground_truth, centroid=centroid, label=label, xy_i=xy_i)
            
            else:
                sc_ground_truth = 0
                nk = 0
                sc = 0
                fm = 0
        
            return image_reconstr_temp, sc_ground_truth, sc, fm, nk, nPhotonR, nPhotonG, nPhotonB
        
        else:
            nk = nCluster
            
            nPhoton = np.zeros(0)
            nScint = 0
            for i in range(nCluster):
                if np.sum(event_label==i) > 0:
                    nScint += 1
                    nPhoton = np.append(nPhoton, np.sum(event_label==i))
            
            if nPhoton > 800:
                sc_ground_truth = silhouette_coeff(int(np.max(event_label)), xy_list, np.ones(xy_list.shape[0]), centroid=source_xy_list, label=event_label)
            
                centroid, label = self.gmeans(xy_list, 20, 20, min_cluster_size=800, max_cluster_size=1500)
                nk = centroid.shape[0]
                
                if not self.verbose:
                    with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
                        log.write('    Gmeans\n')
    
                sc = silhouette_coeff(nk, xy_list, np.ones(RGB_list.size), centroid=centroid, label=label)
                if not self.verbose:
                    with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
                        log.write('    Silhouette Coefficient\n')
                fm = fowlkes_mallows_index(event_label, label)
                if not self.verbose:
                    with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
                        log.write('    Fowlkes Mallows Index\n')
                
                d_mean = np.zeros(nk)
                xrayInc_all = np.zeros((self.energy_tgt.size-1, nk))
            
                energy_distribution_temp = self.energy_distribution.copy()
                energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=0)[np.newaxis,:]
                energy_distribution_temp = energy_distribution_temp/np.sum(energy_distribution_temp, axis=1)[:,np.newaxis]
                energy_distribution_norm = np.zeros((self.energy_distribution.shape[0], self.energy_tgt.size-1))
                for i_tgt in range(self.energy_tgt.size-1):
                    energy_distribution_norm[:,i_tgt] = np.sum(energy_distribution_temp[:,self.energy_tgt_ind[i_tgt]:self.energy_tgt_ind[i_tgt+1]], axis=1)
            
                for k in range(nk):
                    npts = int(np.sum(label==k))
                    if npts > 0:
                        d_mean[k] = np.sum(np.sqrt(np.sum((np.tile(centroid[k,:], (npts,1)) - xy_list[label==k,:])**2, axis=1)))/npts
                    
                        ind_d_mean = np.argmin(np.abs(d_mean[k] - self.d_mean_list))
                        
                        while True:
                            xrayInc = energy_distribution_norm[ind_d_mean,:]
                            # xrayInc = savgol_filter(xrayInc, 10, 5)
                            # xrayInc[xrayInc<0] = 0
                            if np.sum(xrayInc) > 0:
                                # xrayInc /= np.sum(xrayInc)
                                xrayInc_all[:,k] = xrayInc
                                break
                            ind_d_mean -= 1
                    
                        ind_x0 = int((centroid[k,0] - self.dimDetVoxel[0]/2e3)//(self.dimDetVoxel[0]/1e3))
                        x0 = self.dimDetVoxel[0]/2e3 + ind_x0*self.dimDetVoxel[0]/1e3
                        ind_x1 = ind_x0 + 1
                        x1 = self.dimDetVoxel[0]/2e3 + ind_x1*self.dimDetVoxel[0]/1e3
                            
                        ind_y0 = int((centroid[k,1] - self.dimDetVoxel[1]/2e3)//(self.dimDetVoxel[1]/1e3))
                        y0 = self.dimDetVoxel[1]/2e3 + ind_y0*self.dimDetVoxel[1]/1e3
                        ind_y1 = ind_y0 + 1
                        y1 = self.dimDetVoxel[1]/2e3 + ind_y1*self.dimDetVoxel[1]/1e3
            
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
            
                        image_reconstr_temp[ind_x0,ind_y0,:] += xrayInc*ratio[0]
                        image_reconstr_temp[ind_x0,ind_y1,:] += xrayInc*ratio[1]
                        image_reconstr_temp[ind_x1,ind_y0,:] += xrayInc*ratio[2]
                        image_reconstr_temp[ind_x1,ind_y1,:] += xrayInc*ratio[3]
            
                        self.image_reconstr[ind_x0,ind_y0,:] += xrayInc*ratio[0]
                        self.image_reconstr[ind_x0,ind_y1,:] += xrayInc*ratio[1]
                        self.image_reconstr[ind_x1,ind_y0,:] += xrayInc*ratio[2]
                        self.image_reconstr[ind_x1,ind_y1,:] += xrayInc*ratio[3]
                
                if comm.rank == 0:
                    if os.path.exists(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min.npz"):
                        data = np.load(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min.npz")
                        if fm < data['fm']:
                            np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min", d_mean=d_mean, xrayInc=xrayInc_all,
                                     ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, centroid=centroid, xy_list=xy_list, RGB_list=RGB_list,
                                     label=label, source_energy_list=source_energy_list, source_xy_list=source_xy_list, event_label=event_label, sc=sc, fm=fm,
                                     energy_distribution_norm=energy_distribution_norm, sc_ground_truth=sc_ground_truth)
                    else:
                        np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_min", d_mean=d_mean, xrayInc=xrayInc_all,
                                 ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, centroid=centroid, xy_list=xy_list, RGB_list=RGB_list,
                                 label=label, source_energy_list=source_energy_list, source_xy_list=source_xy_list, event_label=event_label, sc=sc, fm=fm,
                                 energy_distribution_norm=energy_distribution_norm, sc_ground_truth=sc_ground_truth)
                    
                    if os.path.exists(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max.npz"):
                        data = np.load(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max.npz")
                        if fm > data['fm']:
                            np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max", d_mean=d_mean, xrayInc=xrayInc_all,
                                     ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, centroid=centroid, xy_list=xy_list, RGB_list=RGB_list,
                                     label=label, source_energy_list=source_energy_list, source_xy_list=source_xy_list, event_label=event_label, sc=sc, fm=fm,
                                     energy_distribution_norm=energy_distribution_norm, sc_ground_truth=sc_ground_truth)
                    else:
                        np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/clustering_pRate_" + str(nk) + "_max", d_mean=d_mean, xrayInc=xrayInc_all,
                                 ind_x0=ind_x0, ind_x1=ind_x1, x0=x0, x1=x1, ind_y0=ind_y0, ind_y1=ind_y1, y0=y0, y1=y1, ratio=ratio, centroid=centroid, xy_list=xy_list, RGB_list=RGB_list,
                                 label=label, source_energy_list=source_energy_list, source_xy_list=source_xy_list, event_label=event_label, sc=sc, fm=fm,
                                 energy_distribution_norm=energy_distribution_norm, sc_ground_truth=sc_ground_truth)
            
            else:
                sc_ground_truth = 0
                nk = 0
                sc = 0
                fm = 0
    
            return image_reconstr_temp, sc_ground_truth, sc, fm, nk, nPhoton
    
    def gmeans(self, xy_list, maxiter_gmeans, maxiter_kmeans, min_cluster_size=1000, max_cluster_size=1000, p_crit=0.05, RGB=False):
        if RGB:
            centroid = np.zeros((3, 5))
            label = np.zeros(xy_list.shape[0])
            for i in range(3):
                ind_i = np.argmax(xy_list[:,2:], axis=1)==i
                xy_i = xy_list[ind_i,:]
                centroid[i,:] = np.mean(xy_i, axis=0)
                label[ind_i] = i
        else:
            centroid = np.mean(xy_list, axis=0).reshape(1, -1)
            label = np.zeros(xy_list.shape[0])
        
        cnt = 0
        while True:
#            with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
#                log.write('\nIteration ' + str(cnt) + '\n')
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
            
            centroid_temp, label_temp = weighted_kmeans(xy_list, np.ones(xy_list.shape[0]), centroid_temp.shape[0], maxiter_kmeans, centroid=centroid_temp)
            
            ind_label = 0
            split_check = False
            centroid = None
            for nC in range(centroid_prev.shape[0]):
                mask_nC = np.isin(label_temp, np.array([2*nC,2*nC+1]))
                
                if np.sum(mask_nC) > min_cluster_size:
                    xy_nC = xy_list[mask_nC,:]
                    label_nC = label_temp[mask_nC]
                
                    # deltaC = centroid_temp[2*nC,:] - centroid_temp[2*nC+1,:]
                    xy_proj = (xy_nC @ eigvec_all[nC,:].reshape(-1, 1)).flatten()/np.sum(eigvec_all[nC,:]**2)
                    xy_proj = (xy_proj - np.mean(xy_proj))/np.std(xy_proj)
                    
    #                result = anderson(xy_proj)
    #                A2star = result.statistic*(1 + 4/xy_proj.size - 25/xy_proj.size**2)
                    
#                    if xy_proj.size == 0:
#                        np.savez(directory + "/data/debug_diptest", centroid=centroid, label=label, centroid_temp=centroid_temp,
#                                 centroid_prev=centroid_prev, xy_proj=xy_proj, pval=pval, label_prev=label_prev, ind_label=ind_label, xy_list=xy_list,
#                                 label_temp=label_temp, xy_nC=xy_nC, label_nC=label_nC, nC=nC)
#                        assert False
                    statistic, pval = diptest.diptest(xy_proj)
                    
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
                    
#                    with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
#                        log.write('Cluster ' + str(nC) + ': ' + str(pval) + ' / ' + str(np.sum(label_temp==2*nC)) + ' / ' + str(np.sum(label_temp==2*nC+1)) + '\n')

            centroid, label = weighted_kmeans(xy_list, np.ones(xy_list.shape[0]), centroid.shape[0], maxiter_kmeans, centroid=centroid)
            fm = fowlkes_mallows_index(label_prev, label)
            
#            if comm.rank == 0:
#            if cnt > 50:
#                np.savez(directory + "/data/debug_gmeans_" + str(cnt) + "_rank" + str(comm.rank), centroid=centroid, label=label, eigval=eigval, eigvec=eigvec, centroid_temp=centroid_temp,
#                         centroid_prev=centroid_prev, deltaC=deltaC, xy_proj=xy_proj, pval=pval, label_prev=label_prev, ind_label=ind_label, xy_list=xy_list,
#                         label_temp=label_temp, xy_nC=xy_nC, label_nC=label_nC)
                
            cnt += 1
            
            if not split_check or cnt > maxiter_gmeans or fm == 1:
                break
        
        return centroid, label
    
    def generate_energy_distribution_data(self, Nruns_per_energy):
        if os.path.exists(directory + "/data/energy_distribution_" + str(comm.rank) + ".npz"):
            os.remove(directory + "/data/energy_distribution_" + str(comm.rank) + ".npz")
        
        source_xy = np.zeros(2)
        
        quo, rem = divmod(self.energy_bins, comm.size)
        data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)])
        data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
        energy_per_proc = self.energy_list[data_disp[comm.rank]:data_disp[comm.rank+1]]
        
        if self.RGB_scintillator:
            energy_distribution_per_proc = np.zeros((3, self.d_mean_bins, self.energy_bins))
            #radius_vs_Nphoton_per_proc = np.zeros((3, self.energy_bins, Nruns_per_energy, 2))
        else:
            energy_distribution_per_proc = np.zeros((self.d_mean_bins, self.energy_bins))
            #radius_vs_Nphoton_per_proc = np.zeros((self.energy_bins, Nruns_per_energy, 2))
        
        for nd in range(data_size[comm.rank]):
            self.make_simulation_macro(Nruns_per_energy, source_energy=energy_per_proc[nd])
            wvl_dict, xy_dict, _, _ = self.read_output(Nruns_per_energy)
            
            for nr in range(Nruns_per_energy):
                if wvl_dict[nr].size > 2:
                    RGB_list = self.RGB_readings(wvl_dict[nr])

                    if self.RGB_scintillator:
#                        energy_distribution_per_proc[0,ind_d_mean,int(data_disp[comm.rank]+nd)] += np.sum(RGB_list[:,0])/np.sum(RGB_list)
#                        energy_distribution_per_proc[1,ind_d_mean,int(data_disp[comm.rank]+nd)] += np.sum(RGB_list[:,1])/np.sum(RGB_list)
#                        energy_distribution_per_proc[2,ind_d_mean,int(data_disp[comm.rank]+nd)] += np.sum(RGB_list[:,2])/np.sum(RGB_list)
                        
                        # energy_distribution_per_proc[ind_d_meanR,ind_d_meanG,ind_d_meanB,int(data_disp[comm.rank]+nd)] += 1
                        
                        RGB_list_raw = RGB_list.copy()
                        xy_list = xy_dict[nr][np.sum(RGB_list_raw>0.1, axis=1)==1,:]
                        RGB_list = RGB_list_raw[np.sum(RGB_list_raw>0.1, axis=1)==1,:]
                        
                        if RGB_list.shape[0] > 2:
                            ind_RGB = np.argmax(np.sum(RGB_list, axis=0))
                            ind_d_mean, d_mean = self.data_interpretation(xy_list) #, np.sum(RGB_list, axis=1)/np.sum(RGB_list, axis=1))
                            energy_distribution_per_proc[ind_RGB,ind_d_mean,int(data_disp[comm.rank]+nd)] += 1
                            #radius_vs_Nphoton_per_proc[ind_RGB,int(data_disp[comm.rank]+nd),nr,:] = np.array([RGB_list.shape[0],d_mean])
                        
                    else:
                        ind_d_mean, d_mean = self.data_interpretation(xy_dict[nr]) #, np.sum(RGB_list, axis=1)/np.sum(RGB_list, axis=1))
                        energy_distribution_per_proc[ind_d_mean,int(data_disp[comm.rank]+nd)] += 1
                        #radius_vs_Nphoton_per_proc[int(data_disp[comm.rank]+nd),nr,:] = np.array([RGB_list.shape[0],d_mean])
        
        np.savez(directory + "/data/energy_distribution_" + str(comm.rank), energy_distribution=energy_distribution_per_proc, #radius_vs_Nphoton=radius_vs_Nphoton_per_proc,
                 wvl_list=wvl_dict[nr], xy_list=xy_dict[nr], energy_list=energy_per_proc)
        
        while True:
            filecheck = np.zeros(comm.size)
            for nproc in range(comm.size):
                filecheck[nproc] = os.path.exists(directory + "/data/energy_distribution_" + str(nproc) + ".npz")
            if np.sum(filecheck) == comm.size:
                break
            time.sleep(1)
        
        time.sleep(5)
        
        if self.energy_distribution is None:
            if self.RGB_scintillator:
                self.energy_distribution = np.zeros((3, self.d_mean_bins, self.energy_bins))
                #self.radius_vs_Nphoton = np.zeros((3, self.energy_bins, Nruns_per_energy, 2))
            else:
                self.energy_distribution = np.zeros((self.d_mean_bins, self.energy_bins))
                #self.radius_vs_Nphoton = np.zeros((self.energy_bins, Nruns_per_energy, 2))
        
        for nproc in range(comm.size):
            data_proc = np.load(directory + "/data/energy_distribution_" + str(nproc) + ".npz")
            self.energy_distribution += data_proc['energy_distribution']
            #self.radius_vs_Nphoton += data_proc['radius_vs_Nphoton']
        
#        if self.RGB_scintillator:
#            radius_vs_Nphoton_processed = np.zeros((3, self.energy_bins, Nruns_per_energy, 2))
#            for i in range(3):
#                for ne in range(self.energy_bins):
#                    cnt = 0
#                    for nph in range(3, int(np.max(self.radius_vs_Nphoton[i,ne,:,0]))):
#                        mask = self.radius_vs_Nphoton[i,ne,:,0] == nph
#                        radius_vs_Nphoton_processed[i,ne,cnt:cnt+int(np.sum(mask)),0] = nph
#                        radius_vs_Nphoton_processed[i,ne,cnt:cnt+int(np.sum(mask)),1] = self.radius_vs_Nphoton[i,ne,mask,1]
#                        cnt += int(np.sum(mask))
        
        if comm.rank == 0:
            np.savez(directory + "/data/energy_distribution_data_" + self.distribution_datafile, energy_distribution=self.energy_distribution)#, radius_vs_Nphoton=self.radius_vs_Nphoton,
                     #radius_vs_Nphoton_processed=radius_vs_Nphoton_processed)
        
        comm.Barrier()
        if comm.rank == 0:
            sync_check = 1
        else:
            sync_check = None
        sync_check = comm.bcast(sync_check, root=0)
    
    def generate_test_image(self, Nphoton_rate, Nimages, imed_sigma, imaging_mode=False):
        if not imaging_mode:
            if self.verbose:
                print('\n### Reconstructing Images | Photon Rate: ' + str(Nphoton_rate), flush=True)
            else:
                with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'w') as log:
                    log.write('\n### Reconstructing Images | Photon Rate: ' + str(Nphoton_rate) + '\n')
        
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
        else:
            with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
                log.write('    Geant4: ' + str(t2-t1) + '\n')
        
        # Initialize Metric Arrays
        if not imaging_mode:
            max_imed_per_proc = np.zeros((imed_sigma.size, data_size[comm.rank]))
            sc_ground_truth_per_proc = np.zeros(data_size[comm.rank])
            silhouette_coeff_per_proc = np.zeros(data_size[comm.rank])
            fowlkes_mallows_index_per_proc = np.zeros(data_size[comm.rank])
            nCluster_per_proc = np.zeros(data_size[comm.rank])
            if self.RGB_scintillator:
                nRPhoton_per_proc = np.zeros(0)
                nGPhoton_per_proc = np.zeros(0)
                nBPhoton_per_proc = np.zeros(0)
            else:
                nPhoton_per_proc = np.zeros(0)
        
        if imaging_mode and self.verbose:
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
            if imaging_mode and self.verbose:
                if ni%(int(data_size[comm.rank]/100)) == 0:
                    print('|', end='', flush=True)
        
            wvl_list = np.zeros(0)
            x_list = np.zeros(0)
            y_list = np.zeros(0)
            event_label = np.zeros(0)
            source_energy_list = np.zeros(Nphoton_rate)
            source_xy_list = np.zeros((Nphoton_rate, 2))
        
            image_true_temp = np.zeros_like(self.image_true).astype(np.float32)
            
#            cnt = 0
#            for nph in range(10*Nphoton_rate):
#                if wvl_dict[nph].size > 0:
#                    source_energy_list[cnt] = energySrc_list[nph]
#                    source_xy_list[cnt,:] = xySrc_list[nph,:] # mm
#                    
#                    ind_x0 = int((xySrc_list[nph,0] - self.dimDetVoxel[0]/2e3)//(self.dimDetVoxel[0]/1e3))
#                    x0 = self.dimDetVoxel[0]/2e3 + ind_x0*self.dimDetVoxel[0]/1e3
#                    ind_x1 = ind_x0 + 1
#                    x1 = self.dimDetVoxel[0]/2e3 + ind_x1*self.dimDetVoxel[0]/1e3
#                    if np.abs(x0 - xySrc_list[nph,0]) <= np.abs(x1 - xySrc_list[nph,0]):
#                        source_x_ind = ind_x0
#                    else:
#                        source_x_ind = ind_x1
#                        
#                    ind_y0 = int((xySrc_list[nph,1] - self.dimDetVoxel[1]/2e3)//(self.dimDetVoxel[1]/1e3))
#                    y0 = self.dimDetVoxel[1]/2e3 + ind_y0*self.dimDetVoxel[1]/1e3
#                    ind_y1 = ind_y0 + 1
#                    y1 = self.dimDetVoxel[1]/2e3 + ind_y1*self.dimDetVoxel[1]/1e3
#                    if np.abs(y0 - xySrc_list[nph,1]) <= np.abs(y1 - xySrc_list[nph,1]):
#                        source_y_ind = ind_y0
#                    else:
#                        source_y_ind = ind_y1
#                    
#                    source_energy_ind = np.argmin(np.abs(self.energy_list - energySrc_list[nph]))
#            
#                    image_true_temp[source_x_ind,source_y_ind,source_energy_ind] += 1
#                    self.image_true[source_x_ind,source_y_ind,source_energy_ind] += 1
#
#                    wvl_list = np.hstack((wvl_list, wvl_dict[nph]))
#                    x_list = np.hstack((x_list, xy_dict[nph][:,0]))
#                    y_list = np.hstack((y_list, xy_dict[nph][:,1]))
#                    event_label = np.hstack((event_label, nph*np.ones(wvl_dict[nph].size)))
#                    
#                    cnt += 1
#                
#                if cnt >= Nphoton_rate:
#                    break
            
            t1 = time.time()
            for nph in range(Nphoton_rate):
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
                
                # source_energy_ind = np.argmin(np.abs(self.energy_list - energySrc_list[nph]))
                for i_tgt in range(self.energy_tgt.size-1):
                    if energySrc_list[ni*Nphoton_rate+nph] >= self.energy_tgt[i_tgt] and energySrc_list[ni*Nphoton_rate+nph] < self.energy_tgt[i_tgt+1]:
                        source_energy_ind = i_tgt
                        break
        
                image_true_temp[source_x_ind,source_y_ind,source_energy_ind] += 1
                self.image_true[source_x_ind,source_y_ind,source_energy_ind] += 1

                wvl_list = np.hstack((wvl_list, wvl_dict[ni*Nphoton_rate+nph]))
                x_list = np.hstack((x_list, xy_dict[ni*Nphoton_rate+nph][:,0]))
                y_list = np.hstack((y_list, xy_dict[ni*Nphoton_rate+nph][:,1]))
                event_label = np.hstack((event_label, nph*np.ones(wvl_dict[ni*Nphoton_rate+nph].size)))
            t2 = time.time()
            if not imaging_mode:
                if self.verbose:
                    print('    True Image: ' + str(t2-t1), flush=True)
                else:
                    with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
                        log.write('    True Image: ' + str(t2-t1) + '\n')
            
            xy_list = np.vstack((x_list, y_list)).T
            
            RGB_list = self.RGB_readings(wvl_list)
            if not self.RGB_scintillator:
                RGB_list = np.sum(RGB_list, axis=1)
            
            if wvl_list.size > 0:
                t1 = time.time()
                if self.RGB_scintillator:
                    image_reconstr_temp, sc_ground_truth_temp, sc_temp, fm_temp, nCluster_temp, nPhotonR_temp, nPhotonG_temp, nPhotonB_temp = self.reconstruct_image(source_energy_list,
                                                                                                                                                                     source_xy_list,
                                                                                                                                                                     xy_list,
                                                                                                                                                                     RGB_list,
                                                                                                                                                                     event_label,
                                                                                                                                                                     nCluster=Nphoton_rate)
                else:
                    image_reconstr_temp, sc_ground_truth_temp, sc_temp, fm_temp, nCluster_temp, nPhoton_temp = self.reconstruct_image(source_energy_list,
                                                                                                                                      source_xy_list,
                                                                                                                                      xy_list,
                                                                                                                                      RGB_list,
                                                                                                                                      event_label,
                                                                                                                                      nCluster=Nphoton_rate)
                t2 = time.time()
                if not imaging_mode:
                    if self.verbose:
                        print('    Reconstruction: ' + str(t2-t1), flush=True)
                    else:
                        with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
                            log.write('    Reconstruction: ' + str(t2-t1) + '\n')
            
                if not imaging_mode:
                    t1 = time.time()
                    max_imed_per_proc[:,ni] = self.compute_imed(image_true_temp, image_reconstr_temp, imed_sigma, Nphoton_rate)
                    t2 = time.time()
                    if self.verbose:
                        print('    IMED: ' + str(t2-t1), flush=True)
                    else:
                        with open(directory + "/logs/Rank" + str(comm.rank) + ".txt", 'a') as log:
                            log.write('    IMED: ' + str(t2-t1) + '\n')
                    sc_ground_truth_per_proc[ni] = sc_ground_truth_temp
                    silhouette_coeff_per_proc[ni] = sc_temp
                    fowlkes_mallows_index_per_proc[ni] = fm_temp
                    nCluster_per_proc[ni] = nCluster_temp
                    if self.RGB_scintillator:
                        nRPhoton_per_proc = np.hstack((nRPhoton_per_proc, nPhotonR_temp[nPhotonR_temp!=0]))
                        nGPhoton_per_proc = np.hstack((nGPhoton_per_proc, nPhotonG_temp[nPhotonG_temp!=0]))
                        nBPhoton_per_proc = np.hstack((nBPhoton_per_proc, nPhotonB_temp[nPhotonB_temp!=0]))
                    else:
                        nPhoton_per_proc = np.hstack((nPhoton_per_proc, nPhoton_temp[nPhoton_temp!=0]))
            else:
                image_reconstr_temp = np.zeros_like(self.image_reconstr).astype(np.float32)
        
        if imaging_mode:
            if self.verbose:
                print('\n    Image Gathering Progress:', flush=True)
                print('    ', end='', flush=True)
        
            if not os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate)):
                os.mkdir(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate))
            if comm.rank != 0:
                np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate) + "/rank" + str(comm.rank),
                         image_true=self.image_true, image_reconstr=self.image_reconstr)
        
            if comm.rank == 0:
                for nr in range(1, comm.size):
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
                
                image_true_norm = self.image_true.copy()
                image_reconstr_norm = self.image_reconstr.copy()
                for ne in range(self.energy_tgt.size-1):
                    image_true_norm[:,:,ne] /= (self.NDetVoxel[0]*self.NDetVoxel[1]*np.sum(self.xraySrc[self.energy_tgt_ind[ne]:self.energy_tgt_ind[ne+1]]))
                    image_reconstr_norm[:,:,ne] /= (self.NDetVoxel[0]*self.NDetVoxel[1]*np.sum(self.xraySrc[self.energy_tgt_ind[ne]:self.energy_tgt_ind[ne+1]]))
                image_true5 = self.compress_image(image_true_norm, 5)
                image_reconstr5 = self.compress_image(image_reconstr_norm, 5)
                image_true10 = self.compress_image(image_true_norm, 10)
                image_reconstr10 = self.compress_image(image_reconstr_norm, 10)
                image_true20 = self.compress_image(image_true_norm, 20)
                image_reconstr20 = self.compress_image(image_reconstr_norm, 20)
                
                np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate), image_true=self.image_true, image_reconstr=self.image_reconstr,
                         image_true5=image_true5, image_reconstr5=image_reconstr5, image_true10=image_true10, image_reconstr10=image_reconstr10, image_true20=image_true20,
                         image_reconstr20=image_reconstr20, image_true_norm=image_true_norm, image_reconstr_norm=image_reconstr_norm)
        
        else:
            if not os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate)):
                os.mkdir(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate))
            if comm.rank != 0:
                if self.RGB_scintillator:
                    np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate) + "/rank" + str(comm.rank),
                             nRPhoton_per_proc=nRPhoton_per_proc, nGPhoton_per_proc=nGPhoton_per_proc, nBPhoton_per_proc=nBPhoton_per_proc)
                else:
                    np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate) + "/rank" + str(comm.rank),
                             nPhoton_per_proc=nPhoton_per_proc)
        
            if self.RGB_scintillator:
                nRPhoton = nRPhoton_per_proc.copy()
                nGPhoton = nGPhoton_per_proc.copy()
                nBPhoton = nBPhoton_per_proc.copy()
            else:
                nPhoton = nPhoton_per_proc.copy()
        
            if comm.rank == 0:
                for nr in range(1, comm.size):
                    while True:
                        if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate) + "/rank" + str(nr) + ".npz"):
                            try:
                                if self.RGB_scintillator:
                                    with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate) + "/rank" + str(nr) + ".npz") as data:
                                        nRPhoton = np.hstack((nRPhoton, data['nRPhoton_per_proc']))
                                        nGPhoton = np.hstack((nGPhoton, data['nGPhoton_per_proc']))
                                        nBPhoton = np.hstack((nBPhoton, data['nBPhoton_per_proc']))
                                else:
                                    with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate) + "/rank" + str(nr) + ".npz") as data:
                                        nPhoton = np.hstack((nPhoton, data['nPhoton_per_proc']))
                                break
                            except:
                                pass
                        else:
                            time.sleep(1)
                    os.remove(directory + "/results/image_reconstruction/" + self.identifier + "/nPhoton_pRate" + str(Nphoton_rate) + "/rank" + str(nr) + ".npz")
        
            comm.Barrier()
        
            max_imed = np.zeros(imed_sigma.size*Nimages)
            sc_ground_truth = np.zeros(Nimages)
            silhouette_coeff = np.zeros(Nimages)
            fowlkes_mallows_index = np.zeros(Nimages)
            nCluster = np.zeros(Nimages)
            
            data_size_imed = data_size*imed_sigma.size
            data_disp_imed = np.array([sum(data_size_imed[:p]) for p in range(size)]).astype(np.float64)
            comm.Gatherv(max_imed_per_proc.T.reshape(-1), [max_imed, data_size_imed, data_disp_imed, MPI.DOUBLE], root=0)
            max_imed = max_imed.reshape(Nimages, imed_sigma.size).T
            
            comm.Gatherv(sc_ground_truth_per_proc, [sc_ground_truth, data_size, data_disp, MPI.DOUBLE], root=0)
            comm.Gatherv(silhouette_coeff_per_proc, [silhouette_coeff, data_size, data_disp, MPI.DOUBLE], root=0)
            comm.Gatherv(fowlkes_mallows_index_per_proc, [fowlkes_mallows_index, data_size, data_disp, MPI.DOUBLE], root=0)
            comm.Gatherv(nCluster_per_proc, [nCluster, data_size, data_disp, MPI.DOUBLE], root=0)
    
            if comm.rank == 0:
                if self.RGB_scintillator:
                    np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/metrics_pRate_" + str(Nphoton_rate), max_imed=max_imed,
                             silhouette_coeff=silhouette_coeff, fowlkes_mallows_index=fowlkes_mallows_index, nCluster=nCluster, nRPhoton=nRPhoton, nGPhoton=nGPhoton, nBPhoton=nBPhoton,
                             sc_ground_truth=sc_ground_truth)
                else:
                    np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/metrics_pRate_" + str(Nphoton_rate), max_imed=max_imed,
                             silhouette_coeff=silhouette_coeff, fowlkes_mallows_index=fowlkes_mallows_index, nCluster=nCluster, nPhoton=nPhoton,
                             sc_ground_truth=sc_ground_truth)
        
        comm.Barrier()
    
    def compute_imed(self, image_true, image_reconstr, sigma, Nphoton_rate): # image Euclidean distance
        dist = np.zeros((sigma.size, self.energy_tgt_index_imed.size))
        
        max_true = np.max(image_true)
        min_true = np.min(image_true)
        max_reconstr = np.max(image_reconstr)
        min_reconstr = np.min(image_reconstr)
        
        if max_true - min_true > 0:
            norm_true = (image_true - min_true)/(max_true - min_true)
        else:
            norm_true = image_true.copy()
        if max_reconstr - min_reconstr > 0:
            norm_reconstr = (image_reconstr - min_reconstr)/(max_reconstr - min_reconstr)
        else:
            norm_reconstr = image_reconstr.copy()
        
        for s in range(sigma.size):
            for n in range(self.energy_tgt_index_imed.size):
                index = self.energy_tgt_index_imed[n]
                filtered_true = gaussian_filter(norm_true[:,:,index], sigma[s], mode='reflect')
                filtered_reconstr = gaussian_filter(norm_reconstr[:,:,index], sigma[s], mode='reflect')
        
                dist[s,n] = np.sqrt(np.sum((filtered_true - filtered_reconstr)**2))
            
                if comm.rank == 0:
                    if np.sum(image_true[:,:,index]) > 0:
                        np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/imed_pRate_" + str(Nphoton_rate) + "_n" + str(n) + "_sigma" + str(sigma[s]),
                                 filtered_true=filtered_true, filtered_reconstr=filtered_reconstr, dist=dist, image_true=image_true[:,:,index], image_reconstr=image_reconstr[:,:,index],
                                 norm_true=norm_true[:,:,index], norm_reconstr=norm_reconstr[:,:,index])
#            if dist[n] > 1 or np.isnan(dist[n]):
#                np.savez(directory + "/data/reconstruction_examples/" + self.identifier + "/imed_debug" + str(np.random.randint(1000)), filtered_true=filtered_true, filtered_reconstr=filtered_reconstr,
#                         dist=dist, image_true=image_true[:,:,n], image_reconstr=image_reconstr[:,:,n], slice_true=slice_true, slice_reconstr=slice_reconstr)
        
        return np.max(dist, axis=1)
    
    def compress_image(self, image, factor):
        image_new = np.sum(image.reshape(image.shape[0]//factor, factor, image.shape[1]//factor, factor, image.shape[2]), axis=(1,3))
        
        return image_new
    
    def plot_trends(self, Nphoton_rate_list, Nimages, imed_sigma):
        Nrate = Nphoton_rate_list.size
    
        max_imed = np.zeros((imed_sigma.size, Nimages, Nrate))
        sc_ground_truth = np.zeros((Nimages, Nrate))
        silhouette_coeff = np.zeros((Nimages, Nrate))
        fowlkes_mallows_index = np.zeros((Nimages, Nrate))
        nCluster = np.zeros((Nimages, Nrate))
        if self.RGB_scintillator:
            nRPhoton = {}
            nGPhoton = {}
            nBPhoton = {}
        else:
            nPhoton = np.zeros((Nimages, Nrate))

        for nph in range(Nrate):
            if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/metrics_pRate_" + str(Nphoton_rate_list[nph]) + ".npz"):
                data = np.load(directory + "/results/image_reconstruction/" + self.identifier + "/metrics_pRate_" + str(Nphoton_rate_list[nph]) + ".npz")
                max_imed[:,:,nph] = data['max_imed']
                sc_ground_truth[:,nph] = data['sc_ground_truth']
                silhouette_coeff[:,nph] = data['silhouette_coeff']
                fowlkes_mallows_index[:,nph] = data['fowlkes_mallows_index']
                nCluster[:,nph] = data['nCluster']
                if self.RGB_scintillator:
                    nRPhoton[nph] = data['nRPhoton']
                    nGPhoton[nph] = data['nGPhoton']
                    nBPhoton[nph] = data['nBPhoton']
                else:
                    nPhoton[nph] = data['nPhoton']
            else:
                break
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[16,16], sharex=True, dpi=100)
        
        marker_list = ['o','^','s','D','p','*']
        for ns in range(imed_sigma.size):
            lower_err = np.min(max_imed[ns,:,:], axis=0)
            upper_err = np.max(max_imed[ns,:,:], axis=0)
            mean_val = np.mean(max_imed[ns,:,:], axis=0)
            errorbar = [mean_val-lower_err,upper_err-mean_val]
            ax1.errorbar(Nphoton_rate_list, mean_val, yerr=errorbar, ecolor='gold', elinewidth=1.5, color='goldenrod', marker=marker_list[ns], linewidth=2)
        
        lower_err = np.min(silhouette_coeff, axis=0)
        upper_err = np.max(silhouette_coeff, axis=0)
        mean_val = np.mean(silhouette_coeff, axis=0)
        errorbar = [mean_val-lower_err,upper_err-mean_val]
        ax2.errorbar(Nphoton_rate_list, mean_val, yerr=errorbar, ecolor='forestgreen', elinewidth=1.5, color='darkgreen', marker='o', linewidth=2)
        
        lower_err = np.min(fowlkes_mallows_index, axis=0)
        upper_err = np.max(fowlkes_mallows_index, axis=0)
        mean_val = np.mean(fowlkes_mallows_index, axis=0)
        errorbar = [mean_val-lower_err,upper_err-mean_val]
        ax3.errorbar(Nphoton_rate_list, mean_val, yerr=errorbar, ecolor='lightskyblue', elinewidth=1.5, color='royalblue', marker='o', linewidth=2)
        plt.savefig(directory + "/plots/" + self.identifier + "_metric_trends.png", dpi=100)
        plt.close()
        
        for nph in range(Nrate):
            if np.sum(nCluster[:,nph]) > 0:
                fig, ax = plt.subplots(1, 1, figsize=[16,12], dpi=100)
                ax.hist(nCluster[:,nph], int(np.max(nCluster[:,nph]) - np.min(nCluster[:,nph])))
                plt.savefig(directory + "/plots/" + self.identifier + "_nCluster_pRate" + str(Nphoton_rate_list[nph]) + ".png", dpi=100)
                plt.close()
                
                if self.RGB_scintillator:
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[16,16], dpi=100)
                    ax1.hist(nRPhoton[nph], int(np.max(nRPhoton[nph]) - np.min(nRPhoton[nph])))
                    ax2.hist(nGPhoton[nph], int(np.max(nGPhoton[nph]) - np.min(nGPhoton[nph])))
                    ax3.hist(nBPhoton[nph], int(np.max(nBPhoton[nph]) - np.min(nBPhoton[nph])))
                    plt.savefig(directory + "/plots/" + self.identifier + "_nPhoton_pRate" + str(Nphoton_rate_list[nph]) + ".png", dpi=100)
                    plt.close()
                else:
                    fig, ax = plt.subplots(1, 1, figsize=[16,12], dpi=100)
                    ax.hist(nPhoton[nph], int(np.max(nPhoton[nph]) - np.min(nPhoton[nph])))
                    plt.savefig(directory + "/plots/" + self.identifier + "_nPhoton_pRate" + str(Nphoton_rate_list[nph]) + ".png", dpi=100)
                    plt.close()
            else:
                break
        
        color_list = ['dimgray','maroon','red','orange','goldenrod','yellowgreen','green','mediumturquoise','deepskyblue','royalblue','blueviolet','violet']
        for nph in range(Nrate):
            if np.sum(sc_ground_truth[:,nph]) > 0:
                fig, ax = plt.subplots(1, 1, figsize=[16,16], dpi=100)
                ax.scatter(sc_ground_truth[:,nph], fowlkes_mallows_index[:,nph], c=color_list[nph%len(color_list)])
            else:
                break
        plt.savefig(directory + "/plots/" + self.identifier + "_sc_ground_truth_vs_fm.png", dpi=100)
        plt.close()
        
        for nph in range(Nrate):
            if np.sum(sc_ground_truth[:,nph]) > 0:
                fig, ax = plt.subplots(1, 1, figsize=[16,16], dpi=100)
                ax.scatter(sc_ground_truth[:,nph], max_imed[0,:,nph], c=color_list[nph%len(color_list)])
            else:
                break
        plt.savefig(directory + "/plots/" + self.identifier + "_sc_ground_truth_vs_imed.png", dpi=100)
        plt.close()
        
        np.savez(directory + "/results/" + self.identifier + "_metric_trend_data", max_imed=max_imed, silhouette_coeff=silhouette_coeff, fowlkes_mallows_index=fowlkes_mallows_index,
                 nCluster=nCluster, sc_ground_truth=sc_ground_truth)

    def plot_images(self, Nphoton_rate_list):
        Nrate = Nphoton_rate_list.size
        
        if self.sampleID == 1:
            filename = 'AM'
        elif self.sampleID == 2:
            filename = 'AF'
            
        image_true = phantom.phantom(filename, self.dimSampleVoxel[2], self.indSampleVoxel, self.energy_list, self.energy_tgt_ind, self.xraySrc, self.IConc, self.GdConc, self.G4directory)
        np.savez(directory + "/results/image_reconstruction/" + self.identifier + "/image_true", image_true=image_true)

        fig, ax = plt.subplots(1, 1, figsize=[16,12], dpi=300)
        ax.plot(self.energy_list, self.xraySrc, linewidth=2)
        plt.savefig(directory + "/plots/" + self.identifier + "_xraySrc.png", dpi=300)
        plt.close()

        for nph in range(Nrate):
            if os.path.exists(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate_list[nph]) + ".npz"):
                with np.load(directory + "/results/image_reconstruction/" + self.identifier + "/arrays_pRate" + str(Nphoton_rate_list[nph]) + ".npz") as data:
                    image_reconstr_norm1 = data['image_reconstr5'][:,:,1]
                    image_reconstr_norm2 = data['image_reconstr5'][:,:,2]
                    image_reconstr_norm4 = data['image_reconstr5'][:,:,4]
                    image_reconstr_norm5 = data['image_reconstr5'][:,:,5]
            else:
                image_reconstr_norm1 = np.zeros((self.NDetVoxel[0], self.NDetVoxel[1]))
                image_reconstr_norm2 = np.zeros((self.NDetVoxel[0], self.NDetVoxel[1]))
                image_reconstr_norm4 = np.zeros((self.NDetVoxel[0], self.NDetVoxel[1]))
                image_reconstr_norm5 = np.zeros((self.NDetVoxel[0], self.NDetVoxel[1]))
                
            fig, ax = plt.subplots(2, 3, figsize=[30,20], dpi=300)

            vmin_true = np.min(image_true[:,:,1:3])
            vmax_true = np.max(image_true[:,:,1:3])
            vmin_reconstr = np.min(image_reconstr_norm1)
            vmax_reconstr = np.max(image_reconstr_norm1)
            ax[0,0].imshow(np.rot90(image_true[:,:,1]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,0].imshow(np.rot90(image_reconstr_norm1), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_reconstr = np.min(image_reconstr_norm2)
            vmax_reconstr = np.max(image_reconstr_norm2)
            ax[0,1].imshow(np.rot90(image_true[:,:,2]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,1].imshow(np.rot90(image_reconstr_norm2), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_true = np.min(image_true[:,:,2] - image_true[:,:,1])
            vmax_true = np.max(image_true[:,:,2] - image_true[:,:,1])
            vabs_true = np.max((np.abs(vmin_true), np.abs(vmax_true)))
            vmin_reconstr = np.min(image_reconstr_norm2 - image_reconstr_norm1)
            vmax_reconstr = np.max(image_reconstr_norm2 - image_reconstr_norm1)
            vabs_reconstr = np.max((np.abs(vmin_reconstr), np.abs(vmax_reconstr)))
            ax[0,2].imshow(np.rot90(image_true[:,:,2] - image_true[:,:,1]), cmap='coolwarm', vmin=-vabs_true, vmax=vabs_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,2].imshow(np.rot90(image_reconstr_norm2 - image_reconstr_norm1), cmap='coolwarm', vmin=-vabs_reconstr, vmax=vabs_reconstr)
            
            plt.savefig(directory + "/plots/" + self.identifier + "_Iodine_images_5_Nrate" + str(Nphoton_rate_list[nph]) + ".png", dpi=300)
            plt.close()
            
            fig, ax = plt.subplots(2, 3, figsize=[30,20], dpi=300)
            
            vmin_true = np.min(image_true[:,:,4:6])
            vmax_true = np.max(image_true[:,:,4:6])
            vmin_reconstr = np.min(image_reconstr_norm4)
            vmax_reconstr = np.max(image_reconstr_norm4)
            ax[0,0].imshow(np.rot90(image_true[:,:,4]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,0].imshow(np.rot90(image_reconstr_norm4), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_reconstr = np.min(image_reconstr_norm5)
            vmax_reconstr = np.max(image_reconstr_norm5)
            ax[0,1].imshow(np.rot90(image_true[:,:,5]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,1].imshow(np.rot90(image_reconstr_norm5), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_true = np.min(image_true[:,:,5] - image_true[:,:,4])
            vmax_true = np.max(image_true[:,:,5] - image_true[:,:,4])
            vabs_true = np.max((np.abs(vmin_true), np.abs(vmax_true)))
            vmin_reconstr = np.min(image_reconstr_norm5 - image_reconstr_norm4)
            vmax_reconstr = np.max(image_reconstr_norm5 - image_reconstr_norm4)
            vabs_reconstr = np.max((np.abs(vmin_reconstr), np.abs(vmax_reconstr)))
            ax[0,2].imshow(np.rot90(image_true[:,:,5] - image_true[:,:,4]), cmap='coolwarm', vmin=-vabs_true, vmax=vabs_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,2].imshow(np.rot90(image_reconstr_norm5 - image_reconstr_norm4), cmap='coolwarm', vmin=-vabs_reconstr, vmax=vabs_reconstr)
            
            plt.savefig(directory + "/plots/" + self.identifier + "_Gadolinium_images_5_Nrate" + str(Nphoton_rate_list[nph]) + ".png", dpi=300)
            plt.close()
            
            fig, ax = plt.subplots(2, 3, figsize=[30,20], dpi=300)

            vmin_true = np.min(image_true[:,:,1:3])
            vmax_true = np.max(image_true[:,:,1:3])
            vmin_reconstr = np.min(image_reconstr_norm1)
            vmax_reconstr = np.max(image_reconstr_norm1)
            ax[0,0].imshow(np.rot90(image_true[:,:,1]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,0].imshow(np.rot90(image_reconstr_norm1), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_reconstr = np.min(image_reconstr_norm2)
            vmax_reconstr = np.max(image_reconstr_norm2)
            ax[0,1].imshow(np.rot90(image_true[:,:,2]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,1].imshow(np.rot90(image_reconstr_norm2), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_true = np.min(image_true[:,:,2] - image_true[:,:,1])
            vmax_true = np.max(image_true[:,:,2] - image_true[:,:,1])
            vabs_true = np.max((np.abs(vmin_true), np.abs(vmax_true)))
            vmin_reconstr = np.min(image_reconstr_norm2 - image_reconstr_norm1)
            vmax_reconstr = np.max(image_reconstr_norm2 - image_reconstr_norm1)
            vabs_reconstr = np.max((np.abs(vmin_reconstr), np.abs(vmax_reconstr)))
            ax[0,2].imshow(np.rot90(image_true[:,:,2] - image_true[:,:,1]), cmap='coolwarm', vmin=-vabs_true, vmax=vabs_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,2].imshow(np.rot90(image_reconstr_norm2 - image_reconstr_norm1), cmap='coolwarm', vmin=-vabs_reconstr, vmax=vabs_reconstr)
            
            plt.savefig(directory + "/plots/" + self.identifier + "_Iodine_images_20_Nrate" + str(Nphoton_rate_list[nph]) + ".png", dpi=300)
            plt.close()
            
            fig, ax = plt.subplots(2, 3, figsize=[30,20], dpi=300)
            
            vmin_true = np.min(image_true[:,:,4:6])
            vmax_true = np.max(image_true[:,:,4:6])
            vmin_reconstr = np.min(image_reconstr_norm4)
            vmax_reconstr = np.max(image_reconstr_norm4)
            ax[0,0].imshow(np.rot90(image_true[:,:,4]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,0].imshow(np.rot90(image_reconstr_norm4), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_reconstr = np.min(image_reconstr_norm5)
            vmax_reconstr = np.max(image_reconstr_norm5)
            ax[0,1].imshow(np.rot90(image_true[:,:,5]), cmap='gray', vmin=vmin_true, vmax=vmax_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,1].imshow(np.rot90(image_reconstr_norm5), cmap='gray', vmin=vmin_reconstr, vmax=vmax_reconstr)
            
            vmin_true = np.min(image_true[:,:,5] - image_true[:,:,4])
            vmax_true = np.max(image_true[:,:,5] - image_true[:,:,4])
            vabs_true = np.max((np.abs(vmin_true), np.abs(vmax_true)))
            vmin_reconstr = np.min(image_reconstr_norm5 - image_reconstr_norm4)
            vmax_reconstr = np.max(image_reconstr_norm5 - image_reconstr_norm4)
            vabs_reconstr = np.max((np.abs(vmin_reconstr), np.abs(vmax_reconstr)))
            ax[0,2].imshow(np.rot90(image_true[:,:,5] - image_true[:,:,4]), cmap='coolwarm', vmin=-vabs_true, vmax=vabs_true, aspect=self.dimSampleVoxel[1]/self.dimSampleVoxel[0])
            ax[1,2].imshow(np.rot90(image_reconstr_norm5 - image_reconstr_norm4), cmap='coolwarm', vmin=-vabs_reconstr, vmax=vabs_reconstr)
            
            plt.savefig(directory + "/plots/" + self.identifier + "_Gadolinium_images_20_Nrate" + str(Nphoton_rate_list[nph]) + ".png", dpi=300)
            plt.close()

@jit(nopython=True, cache=True)
def pca(xy_list):
    xy_mean = np.sum(xy_list, axis=0)/xy_list.shape[0]
    xy_std = np.zeros(xy_list.shape[1])
    for i in range(xy_list.shape[1]):
        xy_std[i] = xy_list[:,i].std()
        
    xy_norm = (xy_list - xy_mean[np.newaxis,:])/xy_std[np.newaxis,:]
    xy_cov = np.cov(xy_norm, rowvar=False, ddof=1)
    eigval, eigvec = np.linalg.eig(xy_cov)
    
    return np.max(eigval), eigvec[:,np.argmax(eigval)]

@jit(nopython=True, cache=True)
def weighted_kmeans(xy_list, weights, nCluster, maxiter, centroid):
    n_iter = 0
    centroid_prev = None
    
    while True:
        label = np.zeros(xy_list.shape[0])
        for xy in range(xy_list.shape[0]):
            distance = np.sqrt(np.sum((xy_list[xy,:][np.newaxis,:] - centroid)**2, axis=1))
            label[xy] = np.argmin(distance)
        
        centroid_prev = centroid.copy()
        
        for nC in range(nCluster):
            centroid[nC,:] = np.sum(xy_list[label==nC,:]*weights[label==nC][:,np.newaxis], axis=0)/np.sum(weights[label==nC])
            if np.isnan(centroid[nC,:]).any():
                centroid[nC,:] = centroid_prev[nC,:]
        
        n_iter += 1
        if n_iter >= maxiter:
            break
        if np.all(centroid - centroid_prev < 1e-2):
            break
#                print('    Iteration ' + str(n_iter) + ': ' + str(centroid), flush=True)
    
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

if __name__ == '__main__':
    if rank == 0:
        verbose = True
    else:
        verbose = False

    simulation = geant4(G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main",
                        RGB_scintillator = True, #---------------------------------------------------------- Change
                        sampleID = 0, # 0: None, 1:AM, 2:AF #----------------------------------------------- Change
                        dimSampleVoxel = np.array([2.137,8,2.137])/np.array([400,400,1]), # in cm
                        indSampleVoxel = np.array([[0,253], # 0...253 indexing
                                                   [119,189], # 0...222 indexing
                                                   [63,64]]), # 1...128 indexing
                        IConc = 0.005,
                        GdConc = 0.005,
                        dimScint = np.array([1.5,1.5]), # in cm
                        dimDetVoxel = np.array([10,10,0.1]), # in um
                        NDetVoxel = np.array([1280,1280,1]),
                        gapSampleScint = 0.0, # in cm
                        gapScintDet = 0., # in cm
                        Nlayers = 3, #---------------------------------------------------------------------- Change
                        scintillator_material = np.array([2,4,5]), # 1:YAGCe, 2:ZnSeTe, 3:LYSOCe, 4:CsITl, 5:GSOCe #- Change
                        scintillator_thickness = np.array([0.050783,0.012911,0.037092]), # in mm #---------- Change 0.222604,0.313417,0.337213 | 0.076081,0.011322,0.073472
                        check_overlap = 0,
                        energy_range = np.array([10,100]), # in keV
                        energy_tgt = np.array([10,28,33,38,45,50,55,150]), # np.linspace(10, 150, 6), # in keV
                        energy_tgt_index_imed = np.array([1,2,4,5]),
                        energy_bins = 91,
                        d_mean_bins = 100,
                        xray_target_material = 'W',
                        xray_operating_V = 100, # in keV
                        xray_tilt_theta = 12, # in deg
                        xray_filter_material = 'Al',
                        xray_filter_thickness = 0.3, # in cm
                        verbose = verbose,
                        distribution_datafile = 'RGB_100keV_245',  #---------------------------------------------------- Change
                        identifier = 'RGB_100keV_245', #-------------------------------------------------------- Change RGB_150keV_nogap_phantom
                        )
    
    if rank == 0:
        simulation.cmake()
        simulation.make()
    else:
        time.sleep(60)
    
#    if verbose:
#        print('\n### Collecting Energy Distribution Data', flush=True)
#        print('    ', end='', flush=True)
#    for i in range(100): # originally 10
#        if verbose:
#            if i%10 == 0:
#                print(i, end='', flush=True)
#            else:
#                print('/', end='', flush=True)
#        simulation.generate_energy_distribution_data(Nruns_per_energy=10000) # originally 100000
#    if verbose:
#        print(i+1, flush=True)
    
    imaging_mode = False
    Nphoton_rate_list = np.hstack((np.linspace(50, 100, 1), np.linspace(200, 500, 4))).astype(int)
    #np.hstack((np.linspace(10, 100, 10), np.linspace(200, 500, 4))).astype(int) / np.linspace(10, 100, 10).astype(int)
    Nphoton_rate_list_plot = np.hstack((np.linspace(50, 100, 1), np.linspace(200, 500, 4))).astype(int)
    #np.hstack((np.linspace(10, 100, 10), np.linspace(200, 500, 4))).astype(int) / np.linspace(10, 100, 10).astype(int)
    imed_sigma = np.array([5,10,20,30,40,50])
    Nimages = int(1e3)
    for Nphoton_rate in Nphoton_rate_list:
        t1 = time.time()
        simulation.generate_test_image(Nphoton_rate, Nimages, imed_sigma, imaging_mode)
        if comm.rank == 0:
            simulation.plot_trends(Nphoton_rate_list_plot, Nimages, imed_sigma)
        comm.Barrier()
        t2 = time.time()
        if verbose:
            print('    | Time Taken: ' + str(t2-t1) + ' s')
    
#    imaging_mode = True
#    Nphoton_rate_list = np.linspace(10, 100, 2).astype(int)
#    Nphoton_rate_list_plot = np.linspace(10, 100, 2).astype(int)
#    imed_sigma = np.array([5,10,20,30,40,50])
#    Nphoton_total = int(1e7)
#    for Nphoton_rate in Nphoton_rate_list:
#        Nimages = int(Nphoton_total/Nphoton_rate)
#        simulation.image_true *= 0
#        simulation.image_reconstr *= 0
##        with np.load(directory + "/results/image_reconstruction/" + simulation.identifier + "/arrays_pRate" + str(Nphoton_rate) + ".npz") as data: # 8e7 done
##            simulation.image_true = data['image_true']
##            simulation.image_reconstr = data['image_reconstr']
#        for i in range(10):
#            t1 = time.time()
#            if verbose:
#                print('### Run ' + str(i+1) + ' of 10', flush=True)
#            if comm.rank == 0:
#                simulation.plot_images(Nphoton_rate_list_plot)
#            simulation.generate_test_image(Nphoton_rate, Nimages, imed_sigma, imaging_mode)
#            if comm.rank == 0:
#                simulation.plot_images(Nphoton_rate_list_plot)
#            comm.Barrier()
#            t2 = time.time()
#            if verbose:
#                print('    | Time Taken: ' + str(t2-t1) + ' s', flush=True)

#    E_sweep = np.linspace(20, 140, 13)
#    fm_map = np.zeros((E_sweep.size, E_sweep.size))
#    imed_map = np.zeros((E_sweep.size, E_sweep.size))
#    for e1 in range(E_sweep.size):
#        for e2 in range(E_sweep.size):
#            t1 = time.time()
#            simulation.energy_tgt[1] = np.min((E_sweep[e1], E_sweep[e2]))
#            simulation.energy_tgt[2] = np.max((E_sweep[e1], E_sweep[e2]))
#            Nimages = int(1e3)
#            simulation.generate_test_image(10, Nimages)
#            if comm.rank == 0:
#                data = np.load(directory + "/results/image_reconstruction/" + simulation.identifier + "/metrics_pRate_10.npz")
#                fm_map[e1,e2] = np.mean(data['fowlkes_mallows_index'])
#                imed_map[e1,e2] = np.mean(data['max_imed'])
#                np.savez(directory + "/results/energy_bin_sweep", fm_map=fm_map, imed_map=imed_map, E_sweep=E_sweep)
#            t2 = time.time()
#            if verbose:
#                print('    | Time Taken: ' + str(t2-t1) + ' s')