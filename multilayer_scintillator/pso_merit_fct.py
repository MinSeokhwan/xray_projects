import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import autograd.numpy as npa
from scipy.optimize import least_squares, Bounds, minimize
import torch
import subprocess
import uproot
import time
import util.read_txt as txt
import color.color_coordinates as cie
import multilayer_scintillator.xray_spectrum_generator as xray
import multilayer_scintillator.training_data_generator as gen
import multilayer_scintillator.pso_merit_fct_jit as jit_fct
import multilayer_scintillator.train_dnn as dnn

from mpi4py import MPI

comm = MPI.COMM_WORLD

class geant4:
    def __init__(self,
                 G4directory,
                 dimScint,
                 dimDetVoxel,
                 NDetVoxel,
                 gapSampleScint,
                 gapScintDet,
                 Nlayers,
                 scintillator_material,
                 check_overlap,
                 energy_range,
                 energy_bins,
                 energy_list,
                 Nphoton,
                 wvl_bins,
                 source_type,
                 xray_target_material=None,
                 xray_operating_V=None,
                 xray_tilt_theta=None,
                 xray_filter_material=None,
                 xray_filter_thickness=None,
                 FoM_type='N_energy_bins',
                 verbose=False,
                 identifier='',
                 ):
                 
        self.G4directory = G4directory

        # Scintillator
        self.dimScint = dimScint
        self.Nlayers = Nlayers
        self.scintillator_material = scintillator_material
        
        # Detector
        self.dimDetVoxel = dimDetVoxel
        self.NDetVoxel = NDetVoxel
        self.dimDet = NDetVoxel*dimDetVoxel
        
        # World
        self.dimWorld = np.max(np.vstack((dimScint, self.dimDet[:2]/1e4)), axis=0) # cm
        self.gapSampleScint = gapSampleScint
        self.gapScintDet = gapScintDet
        
        self.check_overlap = check_overlap
        
        if energy_list is None:
            self.energy_list = np.linspace(energy_range[0], energy_range[1], energy_bins)
            self.energy_bins = energy_bins
        else:
            self.energy_list = energy_list
            self.energy_bins = energy_list.size
        self.Nphoton = Nphoton
        self.wvl_bins = wvl_bins
        
        self.FoM_type = FoM_type
        self.source_type = source_type
        
        # proc = subprocess.Popen([".", "/home/gridsan/smin/geant4/geant4-v11.1.2-install/share/Geant4/geant4make/geant4make.sh"])
        
        self.sensRGB = txt.read_txt(directory + "/data/rgb_sensitivity")
        self.RGB_basis_done = False
        self.set_RGB_basis()
        
        if self.source_type == 'xray_tube':
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

        elif self.source_type == 'uniform':
            self.xraySrc = np.ones(self.energy_bins)/self.energy_bins
        
        self.bfgs_maxiter = 200
        self.verbose = verbose
        self.identifier = identifier
    
    def cmake(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["cmake", ".."])
        time.sleep(10)
        
    def make(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["make", "-j4"])
        time.sleep(20)
    
    def FoM(self, rank, thickness):
        wvl_hist_temp, wvl_bin_centers = self.RGB_calibration(rank, thickness)
    
        if self.FoM_type == 'N_energy_bins':
            wvl_hist_all = wvl_hist_temp @ np.diag(self.xraySrc)
            
            RGB, sensRGB_interp = self.RGB_compute(wvl_hist_all, wvl_bin_centers)
            
            RGB_tgt = np.array([[1,0,0,0],
                                [0,1,1,0],
                                [0,0,0,1]])
            weight = np.array([[1,1,1,1],
                               [1,1,1,1],
                               [1,1,1,1]])
                        
            fom = np.sum(weight*(RGB - RGB_tgt)**2)
            
#            ratio = np.zeros((2, self.energy_bins))
#            for i in range(self.energy_bins):
#                cnt = 0
#                for j in range(3):
#                    if i != j:
#                        ratio[cnt,i] = RGB[i,i]/RGB[j,i]
#                        cnt += 1
#            fom = np.min(ratio)
        
        if rank == 0:
            if self.FoM_type == 'N_energy_bins':
                np.savez(directory + "/results/RGB_data_" + self.identifier, wvl_hist_all=wvl_hist_all, wvl_bin_centers=wvl_bin_centers,
                         RGB=RGB, fom=fom, sensRGB=sensRGB_interp)
        
        return fom
    
    def set_RGB_basis(self):
        scintillator_names = ['YAGCe','ZnSeTe','LYSOCe','CsITl','GSOCe']
        self.RGB_basis = np.zeros((3, self.scintillator_material.size))
        for nmat in range(self.scintillator_material.size):
            emission = txt.read_txt(directory + "/data/emission" + scintillator_names[self.scintillator_material[nmat]-1])
            RGB_temp, _ = self.RGB_compute(emission[:,1].reshape(-1,1), emission[:,0])
            self.RGB_basis[:,nmat] = RGB_temp[:,0]
        self.RGB_basis_done = True
    
    def RGB_calibration(self, rank, thickness):
        if self.verbose:
            print('\n### Wavelength Conversion Matrix Computation', flush=True)
        wvl_hist_temp, wvl_bin_centers = self.test_all_energies(rank, thickness)
        sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
        for i in range(3):
            sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
            sensRGB_interp[:,i] /= 100
        self.raw_clip = np.max(sensRGB_interp.T @ wvl_hist_temp @ self.xraySrc)
        
        return wvl_hist_temp, wvl_bin_centers
    
    def RGB_compute(self, wvl_hist_all, wvl_bin_centers):
        sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
        for i in range(3):
            sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
            sensRGB_interp[:,i] /= 100
            
        RGB = np.zeros((3, wvl_hist_all.shape[1]))
        for i in range(3):
            for j in range(wvl_hist_all.shape[1]):
                RGB[i,j] = np.trapz(sensRGB_interp[:,i]*wvl_hist_all[:,j], wvl_bin_centers)
        # RGB_unclip = sensRGB_interp.T @ wvl_hist_all
        # RGB /= self.raw_clip
        RGB_original = RGB.copy()
        
        if self.RGB_basis_done:
            RGB = np.linalg.inv(self.RGB_basis) @ RGB
            
        RGB /= np.max(RGB)
        
        if comm.rank == 0:
            np.savez(directory + "/data/RGB_compute", RGB_original=RGB_original, RGB=RGB, RGB_basis=self.RGB_basis)
        
        return RGB, sensRGB_interp
    
    def make_simulation_macro(self, rank, thickness, source_energy):
        dzWorld = 1.01*(self.gapSampleScint + np.sum(thickness)/10 + self.gapScintDet + self.dimDet[2]/1e4)
        
        with open(self.G4directory + "/build/run_detector_rank" + str(rank) + ".mac", "w") as mac:
            mac.write("/system/root_file_name output" + str(rank) + ".root\n")
        
            mac.write("/structure/xWorld " + str(np.round(self.dimWorld[0], 6)) + "\n")
            mac.write("/structure/yWorld " + str(np.round(self.dimWorld[1], 6)) + "\n")
            mac.write("/structure/zWorld " + str(np.round(dzWorld, 6)) + "\n\n")
            mac.write("/structure/gapSampleScint " + str(np.round(self.gapSampleScint, 6)) + "\n")
            mac.write("/structure/gapScintDet " + str(np.round(self.gapScintDet, 6)) + "\n")
            
            mac.write("/structure/xScint " + str(np.round(self.dimScint[0], 6)) + "\n")
            mac.write("/structure/yScint " + str(np.round(self.dimScint[1], 6)) + "\n")
            
            mac.write("/structure/nLayers " + str(int(self.Nlayers)) + "\n")
            
            for nl in range(self.Nlayers):
                mac.write("/structure/scintillatorThickness" + str(nl+1) + " %.6f\n" %(thickness[nl]))
            
            for nl in range(self.Nlayers):
                mac.write("/structure/scintillatorMaterial" + str(nl+1) + " " + str(int(self.scintillator_material[nl])) + "\n")
            mac.write("\n")

            mac.write("/structure/sampleID 0\n")
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
            mac.write("/gps/pos/type Point\n")
            mac.write("/gps/pos/centre 0 0 " + str(np.round(-dzWorld/2.02, 6)) + " cm\n")
            mac.write("/gps/ene/type Mono\n")
            mac.write("/gps/ene/mono " + str(np.round(source_energy, 6)) + " keV\n")
            
            mac.write("/run/beamOn " + str(int(self.Nphoton)))
    
    def read_output(self, rank, n, detector, thickness):
        if os.path.exists(self.G4directory + "/build/output" + str(rank) + ".root"):
             os.remove(self.G4directory + "/build/output" + str(rank) + ".root")
    
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["./NS", "run_detector_rank" + str(rank) + ".mac"])
        
        while True:
            if os.path.exists(self.G4directory + "/build/output" + str(rank) + ".root"):
                break
            time.sleep(1)
        
        while True:
            try:
                root_file = uproot.open(self.G4directory + "/build/output" + str(rank) + ".root")
                photons = root_file['Photons']
                break
            except:
                time.sleep(1)
        
        creatorProcess = [t for t in photons["fCreatorProcess"].array()]
        scint_photon_ind = [t == 'Scintillation' for t in creatorProcess]
        wvl_list = np.array(photons["fWlen"].array())
        wvl_hist, wvl_bin_edges = np.histogram(wvl_list[scint_photon_ind], bins=self.wvl_bins, range=(300,1100))
        wvl_bin_centers = (wvl_bin_edges[:-1] + wvl_bin_edges[1:])/2
        
        return wvl_hist, wvl_bin_centers
    
    def test_all_energies(self, rank, thickness, calibration=False):
        wvl_hist_all = np.zeros((self.wvl_bins, self.energy_bins))
        if calibration:
            n_start = self.energy_bins - 1
        else:
            n_start = 0
            
        for n in range(n_start, self.energy_bins):
            t1 = time.time()
            self.make_simulation_macro(rank, thickness, self.energy_list[n])
        
            wvl_hist, wvl_bin_centers = self.read_output(rank, n, detector=True, thickness=thickness)
            wvl_hist_all[:,n] = wvl_hist
            t2 = time.time()
            if self.verbose:
                print('\t | Energy: ' + str(self.energy_list[n]) + 'keV (' + str(t2 - t1) + ' s)', flush=True)
        
        return wvl_hist_all, wvl_bin_centers

if __name__ == '__main__':
    verbose = True

    identifier = 'energy_bins_20_40_60keV' #'energy_bins_30_35_40keV'
    simulation = geant4(G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main",
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
    
    simulation.cmake()
    simulation.make()
    
    fom = simulation.FoM(0, thickness=np.array([0.050783,0.012911,0.037092])) # in mm 0.076081,0.011322,0.073472
    print("FoM: " + str(fom))
    
    # wvl_hist, wvl_bin = simulation.analyze_all_energies(0, thickness=np.array([0.527273,0.809091,0.184700]), pht_per_energy=simulation.Nphoton*np.ones(simulation.energy_bins))