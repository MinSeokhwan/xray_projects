import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
import subprocess
import uproot
import time
import util.read_txt as txt
import color.color_coordinates as cie
import multilayer_scintillator.xray_spectrum_generator as xray

class geant4:
    def __init__(self,
                 G4directory,
                 dimScint,
                 dimDet,
                 gap,
                 structureID,
                 Nlayers,
                 scintillator_material,
                 max_thickness,
                 Ngrid,
                 detector_thickness,
                 check_overlap,
                 energy_range,
                 energy_bins,
                 Nphoton,
                 wvl_bins,
                 calibration_method,
                 source_type,
                 xray_target_material=None,
                 xray_operating_V=None,
                 xray_tilt_theta=None,
                 xray_filter_material=None,
                 xray_filter_thickness=None,
                 imaging_samples=None,
                 FoM_type='N_energy_bins',
                 ):
                 
        self.G4directory = G4directory
        
        self.dimScint = dimScint
        self.dimDet = dimDet
        self.dimWorld = np.max(np.vstack((dimScint, dimDet)), axis=0)
        self.gap = gap
        
        self.max_thickness = max_thickness
        
        self.structureID = structureID
        self.Nlayers = Nlayers
        self.scintillator_material = scintillator_material
        
        self.Ngrid = Ngrid
        self.detector_thickness = detector_thickness
        
        self.check_overlap = check_overlap
        
        self.energy_list = np.linspace(energy_range[0], energy_range[1], energy_bins)
        self.energy_bins = energy_bins
        self.Nphoton = Nphoton
        self.wvl_bins = wvl_bins
        
        self.calibration_method = calibration_method
        self.calibration_done = False
        self.FoM_type = FoM_type
        self.source_type = source_type
        
        # proc = subprocess.Popen([".", "/home/gridsan/smin/geant4/geant4-v11.1.2-install/share/Geant4/geant4make/geant4make.sh"])
        
        self.sensRGB = txt.read_txt(directory + "/data/rgb_sensitivity")
        
        if self.source_type == 'xray_tube':
            self.energy_list, xraySrc = xray.get_spectrum(xray_target_material,
                                                          self.energy_list,
                                                          xray_operating_V,
                                                          xray_tilt_theta,
                                                          xray_filter_material,
                                                          xray_filter_thickness,
                                                          )
            self.energy_bins = self.energy_list.size
            self.xraySrc = xraySrc/np.sum(xraySrc)
            
            if FoM_type == 'K_edge_imaging':
                self.Nsample = imaging_samples['N']
                self.xraySrc_sample = np.zeros((self.energy_bins, self.Nsample))
                for ns in range(self.Nsample):
                    sample_mat = imaging_samples['sample'+str(ns)+'mat']
                    
                    linAttCoeff = np.zeros(self.energy_bins)
                    for nm in range(sample_mat.size):
                        raw = txt.read_txt(directory + '/data/linAttCoeff' + sample_mat[nm])
                        linAttCoeff += np.interp(self.energy_list, raw[:,0], raw[:,1], left=0, right=0)*imaging_samples['sample'+str(ns)+'frac'][nm]
                        
                    self.xraySrc_sample[:,ns] = xraySrc*np.exp(-linAttCoeff*imaging_samples['sample'+str(ns)+'thickness'])
                
                normalization_sample = np.argmin(np.sum(self.xraySrc_sample, axis=0))
                self.xraySrc_sample /= np.sum(self.xraySrc_sample[:,normalization_sample])
                
                np.savez(directory + '/data/xray_spectrum_per_sample', xraySrc_sample=self.xraySrc_sample, xraySrc=xraySrc, linAttCoeff=linAttCoeff)
            else:
                energy_list_dense, xraySrc_dense = xray.get_spectrum(xray_target_material,
                                                                     np.linspace(1, xray_operating_V, int(xray_operating_V)),
                                                                     xray_operating_V,
                                                                     xray_tilt_theta,
                                                                     xray_filter_material,
                                                                     xray_filter_thickness,
                                                                     )
            
                np.savez(directory + '/data/xray_spectrum', energy_list=self.energy_list, xraySrc=self.xraySrc, energy_list_dense=energy_list_dense, xraySrc_dense=xraySrc_dense)
    
    def cmake(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["cmake", ".."])
        time.sleep(10)
        
    def make(self):
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["make", "-j4"])
        time.sleep(20)
    
    def FoM(self, rank, thickness):
        if self.FoM_type == 'N_energy_bins':
            wvl_hist_all, wvl_bin_centers = self.test_all_energies(rank, thickness)
            # wvl_hist_all, wvl_bin_centers = self.analyze_all_energies(thickness)
            
            if np.max(wvl_hist_all) != 0:
                wvl_hist_all /= np.max(wvl_hist_all)
                
            avg_dist = 0
            for i in range(wvl_hist_all.shape[1]):
                # avg_dist += np.sum(np.sqrt(np.sum((RGB[:,i][:,np.newaxis] - RGB)**2, axis=0)))
                avg_dist += np.sum(np.sqrt(np.sum((wvl_hist_all[:,i][:,np.newaxis] - wvl_hist_all)**2, axis=0)))
                # avg_dist -= np.sum(wvl_hist_all_norm[:,i][:,np.newaxis]*wvl_hist_all_norm[:,np.arange(self.energy_bins)!=i])
            avg_dist /= wvl_hist_all.shape[1]*(wvl_hist_all.shape[1] - 1)
            
        elif self.FoM_type == 'K_edge_imaging':
            wvl_hist_temp, wvl_bin_centers = self.test_all_energies(rank, thickness)
            
            wvl_hist_all = np.zeros((self.wvl_bins, self.Nsample + 1))
            wvl_hist_save = np.zeros((self.wvl_bins, self.energy_bins, self.Nsample))
            for ns in range(self.Nsample):
                wvl_hist_all[:,ns] = wvl_hist_temp @ self.xraySrc_sample[:,ns]
                wvl_hist_save[:,:,ns] = wvl_hist_temp*self.xraySrc_sample[:,ns][np.newaxis,:]
            
            if np.max(wvl_hist_all) != 0:
                wvl_hist_all /= np.max(wvl_hist_all)
            
#            avg_dist = np.sqrt(np.sum((1 - wvl_hist_all[wvl_bin_centers<600,0])**2) + np.sum(wvl_hist_all[wvl_bin_centers>=600,0]**2))/wvl_bin_centers.size
#            avg_dist += np.sqrt(np.sum((1 - wvl_hist_all[wvl_bin_centers>600,1])**2) + np.sum(wvl_hist_all[wvl_bin_centers<=600,1]**2))/wvl_bin_centers.size
            avg_dist = 0
            for i in range(wvl_hist_all.shape[1]):
                avg_dist += np.sum(np.sqrt(np.sum((wvl_hist_all[:,i][:,np.newaxis] - wvl_hist_all)**2, axis=0)))
            avg_dist /= wvl_hist_all.shape[1]*(wvl_hist_all.shape[1] - 1)
        
        if self.calibration_method == 'unity':
            if np.max(wvl_hist_all) != 0:
                wvl_hist_all /= np.max(wvl_hist_all)
            
            if not self.calibration_done:
                self.sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
                for i in range(3):
                    self.sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
                    self.sensRGB_interp[:,i] /= np.sum(self.sensRGB_interp[:,i])
                self.calibration_done = True
                
            RGB = self.sensRGB_interp.T @ wvl_hist_all
        
        elif self.calibration_method == 'photon_count':
#            if not self.calibration_done:
#                self.reference_white_RGB = np.zeros((self.wvl_bins, 3))
#                for i in range(3):
#                    thickness_temp = 1e-6*np.ones(3)
#                    thickness_temp[i] = self.max_thickness
#                    wvl_hist_temp, _ = self.test_all_energies(rank, thickness_temp, self.Nphoton*np.ones(self.energy_bins), calibration=True)
#                    self.reference_white_RGB[:,i] = wvl_hist_temp[:,-1]
#                
            self.sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
            for i in range(3):
                self.sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
                self.sensRGB_interp[:,i] /= 100
#                
#                self.Nph_RGB = np.zeros(3)
#                for i in range(3):
#                    self.Nph_RGB[i] = np.dot(self.sensRGB_interp[:,i], self.reference_white_RGB[:,i])
#                self.Nph_norm = np.min(self.Nph_RGB) #np.sum(self.Nph_RGB)/self.Nph_RGB.size
#                
#                self.calibration_done = True
            
            # RGB_unclip = (self.sensRGB_interp.T @ wvl_hist_all)/self.Nph_norm
            RGB_unclip = self.sensRGB_interp.T @ wvl_hist_all
            RGB = RGB_unclip.copy()
            # RGB[RGB > 1] = 1
            RGB /= np.max(RGB)
            
            if rank == 0:
#                np.savez(directory + '/data/calibration_data', wvl_bin_centers=wvl_bin_centers,
#                         sensRGB=self.sensRGB_interp, Nph_norm=self.Nph_norm, Nph_RGB=self.Nph_RGB,
#                         reference_white_RGB=self.reference_white_RGB, RGB_unclip=RGB_unclip, RGB=RGB, wvl_hist_all=wvl_hist_all, wvl_hist_save=wvl_hist_save)
                if self.FoM_type == 'N_energy_bins':
                    np.savez(directory + '/data/calibration_data', wvl_bin_centers=wvl_bin_centers,
                             sensRGB=self.sensRGB_interp, RGB_unclip=RGB_unclip, RGB=RGB, wvl_hist_all=wvl_hist_all)
                elif self.FoM_type == 'K_edge_imaging':
                    np.savez(directory + '/data/calibration_data', wvl_bin_centers=wvl_bin_centers,
                             sensRGB=self.sensRGB_interp, RGB_unclip=RGB_unclip, RGB=RGB, wvl_hist_all=wvl_hist_all, wvl_hist_save=wvl_hist_save)
        
        if rank == 0:
            if self.FoM_type == 'N_energy_bins':
                np.savez(directory + "/results/RGB_data", wvl_hist_all=wvl_hist_all, wvl_bin_centers=wvl_bin_centers,
                         RGB=RGB, avg_dist=avg_dist, fom_val=fom_val)
            elif self.FoM_type == 'K_edge_imaging':
                np.savez(directory + "/results/RGB_data", wvl_hist_all=wvl_hist_all, wvl_hist_save=wvl_hist_save, wvl_bin_centers=wvl_bin_centers,
                         RGB=RGB, avg_dist=avg_dist)
        
        return avg_dist
    
    def make_simulation_macro(self, rank, thickness, source_energy):
        if os.path.exists(self.G4directory + "/build/output" + str(rank) + ".root"):
             os.remove(self.G4directory + "/build/output" + str(rank) + ".root")
        
        with open(self.G4directory + "/build/run_detector_rank" + str(rank) + ".mac", "w") as mac:
            mac.write("/system/root_file_name output" + str(rank) + ".root\n")
        
            mac.write("/structure/xWorld " + str(np.round(self.dimWorld[0], 6)) + " cm\n")
            mac.write("/structure/yWorld " + str(np.round(self.dimWorld[1], 6)) + " cm\n")
            mac.write("/structure/zWorld " + str(np.round(2*(np.sum(thickness)/10 + self.gap + 5), 6)) + " cm\n\n")
            mac.write("/structure/gap " + str(np.round(self.gap, 6)) + " cm\n")
            
            mac.write("/structure/xScint " + str(np.round(self.dimScint[0], 6)) + " cm\n")
            mac.write("/structure/yScint " + str(np.round(self.dimScint[1], 6)) + " cm\n")
            
            mac.write("/structure/structureID " + str(int(self.structureID)) + "\n")
            mac.write("/structure/nLayers " + str(int(self.Nlayers)) + "\n")
            mac.write("/structure/scintillatorThickness1 %.6f mm\n" %(thickness[0]))
            mac.write("/structure/scintillatorThickness2 %.6f mm\n" %(thickness[1]))
            mac.write("/structure/scintillatorThickness3 %.6f mm\n" %(thickness[2]))
            if self.Nlayers > 3:
                mac.write("/structure/scintillatorThickness4 " + str(thickness[3]) + " mm\n")
            mac.write("/structure/scintillatorMaterial1 " + str(int(self.scintillator_material[0])) + "\n")
            mac.write("/structure/scintillatorMaterial2 " + str(int(self.scintillator_material[1])) + "\n")
            mac.write("/structure/scintillatorMaterial3 " + str(int(self.scintillator_material[2])) + "\n")
            if self.Nlayers > 3:
                mac.write("/structure/scintillatorMaterial4 " + str(int(self.scintillator_material[3])) + "\n")
            mac.write("\n")
            
            # mac.write("/structure/sampleThickness " + str(np.round(self.sample_thickness, 6)) + " cm\n")
            # mac.write("/structure/sampleID " + str(int(sampleID + 1)) + "\n\n")
            
            mac.write("/structure/constructDetectors 1\n")
            mac.write("/structure/xDet " + str(np.round(self.dimDet[0], 6)) + " cm\n")
            mac.write("/structure/xDet " + str(np.round(self.dimDet[1], 6)) + " cm\n")
            mac.write("/structure/nGridX " + str(int(self.Ngrid[0])) + "\n")
            mac.write("/structure/nGridY " + str(int(self.Ngrid[1])) + "\n")
            mac.write("/structure/detectorDepth " + str(np.round(self.detector_thickness, 6)) + " um\n\n")
            
            mac.write("/structure/checkDetectorsOverlaps " + str(int(self.check_overlap)) + "\n\n")
            
            # mac.write("/run/numberOfThreads " + str(int(self.Nthreads)) + "\n")
            mac.write("/run/initialize\n")
            # mac.write("/control/execute vis.mac\n\n")
            
            mac.write("/gun/momentumAmp " + str(np.round(source_energy, 6)) + " keV\n")
            mac.write("/run/beamOn " + str(int(self.Nphoton)))
        
        with open(self.G4directory + "/build/run_scintillator_rank" + str(rank) + ".mac", "w") as mac:
            mac.write("/system/root_file_name output" + str(rank) + ".root\n")
            
            mac.write("/structure/xWorld " + str(np.round(self.dimWorld[0], 6)) + " cm\n")
            mac.write("/structure/yWorld " + str(np.round(self.dimWorld[1], 6)) + " cm\n")
            mac.write("/structure/zWorld " + str(np.round(2*(np.sum(thickness)/10 + self.gap + 5), 6)) + " cm\n\n")
            mac.write("/structure/gap " + str(np.round(self.gap, 6)) + " cm\n")
            
            mac.write("/structure/xScint " + str(np.round(self.dimScint[0], 6)) + " cm\n")
            mac.write("/structure/yScint " + str(np.round(self.dimScint[1], 6)) + " cm\n")
            
            mac.write("/structure/structureID " + str(int(self.structureID)) + "\n")
            mac.write("/structure/nLayers " + str(int(self.Nlayers)) + "\n")
            mac.write("/structure/scintillatorThickness1 %.6f mm\n" %(thickness[0]))
            mac.write("/structure/scintillatorThickness2 %.6f mm\n" %(thickness[1]))
            mac.write("/structure/scintillatorThickness3 %.6f mm\n" %(thickness[2]))
            if self.Nlayers > 3:
                mac.write("/structure/scintillatorThickness4 " + str(thickness[3]) + " mm\n")
            mac.write("/structure/scintillatorMaterial1 " + str(int(self.scintillator_material[0])) + "\n")
            mac.write("/structure/scintillatorMaterial2 " + str(int(self.scintillator_material[1])) + "\n")
            mac.write("/structure/scintillatorMaterial3 " + str(int(self.scintillator_material[2])) + "\n")
            if self.Nlayers > 3:
                mac.write("/structure/scintillatorMaterial4 " + str(int(self.scintillator_material[3])) + "\n")
            mac.write("\n")
            
            # mac.write("/structure/sampleThickness " + str(np.round(self.sample_thickness, 6)) + " cm\n")
            # mac.write("/structure/sampleID " + str(int(sampleID + 1)) + "\n\n")
            
            mac.write("/structure/constructDetectors 0\n")
            mac.write("/structure/xDet " + str(np.round(self.dimDet[0], 6)) + " cm\n")
            mac.write("/structure/xDet " + str(np.round(self.dimDet[1], 6)) + " cm\n")
            mac.write("/structure/nGridX " + str(int(self.Ngrid[0])) + "\n")
            mac.write("/structure/nGridY " + str(int(self.Ngrid[1])) + "\n")
            mac.write("/structure/detectorDepth " + str(np.round(self.detector_thickness, 6)) + " um\n\n")
            
            mac.write("/structure/checkDetectorsOverlaps " + str(int(self.check_overlap)) + "\n\n")
            
            # mac.write("/run/numberOfThreads " + str(int(self.Nthreads)) + "\n")
            mac.write("/run/initialize\n")
            # mac.write("/control/execute vis.mac\n\n")
            
            mac.write("/gun/momentumAmp " + str(np.round(source_energy, 6)) + " keV\n")
            mac.write("/run/beamOn " + str(int(self.Nphoton)))
    
    def read_output(self, rank, n, detector, thickness):
        os.chdir(self.G4directory + "/build")
        if detector:
            proc = subprocess.Popen(["./NS", "run_detector_rank" + str(rank) + ".mac"])
        else:
            proc = subprocess.Popen(["./NS", "run_scintillator_rank" + str(rank) + ".mac"]) 
        
        while True:
            if os.path.exists(self.G4directory + "/build/output" + str(rank) + ".root"):
                break
            time.sleep(1)
        
        file_size = 0
        while True:
            # file_size_new = os.path.getsize(self.G4directory + "/build/output.root")
            # if file_size_new == file_size and file_size_new != 0:
            #     break
            # else:
            #     file_size = file_size_new
            
            try:
                root_file = uproot.open(self.G4directory + "/build/output" + str(rank) + ".root")
                photons = root_file['Photons']
                break
            except:
                time.sleep(1)
        
        # try:
        #     root_file = uproot.open(self.G4directory + "/build/output" + str(rank) + ".root")
        #     photons = root_file['Photons']
        # except:
        #     wvl_bin_edges = np.linspace(300, 1100, self.wvl_bins + 1)
        #     wvl_bin_centers = (wvl_bin_edges[:-1] + wvl_bin_edges[1:])/2
        #     return np.zeros(wvl_bin_centers.size), wvl_bin_centers
        
        creatorProcess = [t for t in photons["fCreatorProcess"].array()]
        scint_photon_ind = [t == 'Scintillation' for t in creatorProcess]
        wvl_list = np.array(photons["fWlen"].array())
        wvl_hist, wvl_bin_edges = np.histogram(wvl_list[scint_photon_ind], bins=self.wvl_bins, range=(300,1100))
        wvl_bin_centers = (wvl_bin_edges[:-1] + wvl_bin_edges[1:])/2
            
        if not detector:
            fMaterial = np.array(photons["fMaterial"].array())
            
            scint_YAGCe = wvl_list[fMaterial == 'YAGCe']
            scint_ZnSeTe = wvl_list[fMaterial == 'ZnSeTe']
            scint_LYSOCe = wvl_list[fMaterial == 'LYSOCe']
            spec_YAGCe, wvl_bin_edges = np.histogram(scint_YAGCe, bins=self.wvl_bins, range=(300,1100))
            spec_ZnSeTe, wvl_bin_edges = np.histogram(scint_ZnSeTe, bins=self.wvl_bins, range=(300,1100))
            spec_LYSOCe, wvl_bin_edges = np.histogram(scint_LYSOCe, bins=self.wvl_bins, range=(300,1100))
            
            z_list = np.array(photons["fZ"].array()) # in mm
            z_scint_YAGCe = z_list[fMaterial == 'YAGCe']
            z_scint_ZnSeTe = z_list[fMaterial == 'ZnSeTe']
            z_scint_LYSOCe = z_list[fMaterial == 'LYSOCe']
            z_YAGCe, z_bin_edges = np.histogram(z_scint_YAGCe, bins=int(np.sum(thickness)/0.001), range=(0,np.sum(thickness)))
            z_ZnSeTe, z_bin_edges = np.histogram(z_scint_ZnSeTe, bins=int(np.sum(thickness)/0.001), range=(0,np.sum(thickness)))
            z_LYSOCe, z_bin_edges = np.histogram(z_scint_LYSOCe, bins=int(np.sum(thickness)/0.001), range=(0,np.sum(thickness)))
            
            if rank == 0:
                np.savez(directory + "/results/analysis_data_energy" + str(int(n)),
                         wvl_hist=wvl_hist, wvl_bin_centers=wvl_bin_centers,
                         spec_YAGCe=spec_YAGCe, spec_ZnSeTe=spec_ZnSeTe,
                         spec_LYSOCe=spec_LYSOCe, z_bin_edges=z_bin_edges,
                         z_YAGCe=z_YAGCe, z_ZnSeTe=z_ZnSeTe, z_LYSOCe=z_LYSOCe)
        
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
        
        return wvl_hist_all, wvl_bin_centers
    
    def analyze_all_energies(self, rank, thickness):
        wvl_hist_all = np.zeros((self.wvl_bins, self.energy_bins))
        for n in range(self.energy_bins):
            t1 = time.time()
            self.make_simulation_macro(rank, thickness, self.energy_list[n])
            
            wvl_hist, wvl_bin_centers = self.read_output(rank, n, detector=False, thickness=thickness)
            wvl_hist_all[:,n] = wvl_hist
            t2 = time.time()
        
        return wvl_hist_all, wvl_bin_centers

if __name__ == '__main__':
    imaging_samples = {}
    imaging_samples['N'] = 2
    imaging_samples['sample0mat'] = np.array(['Bone'])
    imaging_samples['sample0frac'] = np.array([1])
    imaging_samples['sample0thickness'] = 10 # in cm
    imaging_samples['sample1mat'] = np.array(['Blood','I'])
    imaging_samples['sample1frac'] = np.array([0.991,0.009])
    imaging_samples['sample1thickness'] = 10 # in cm

    simulation = geant4(G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main",
                        dimScint = np.array([1,1]), # in cm
                        dimDet = np.array([1,1]), # in cm
                        gap = 10, # in cm
                        structureID = 1,
                        Nlayers = 3,
                        scintillator_material = np.array([2,1,3]), # 1:YAGCe, 2:ZnSeTe, 3:LYSOCe, 4:CsITl
                        max_thickness = 1, # in mm
                        Ngrid = np.array([1,1]),
                        detector_thickness = 0.1, # in um
                        check_overlap = 1,
                        energy_range = np.array([30,100]), # in keV
                        energy_bins = 8,
                        Nphoton = 100000,
                        wvl_bins = 801,
                        calibration_method = 'photon_count',
                        source_type = 'uniform',
                        xray_target_material = 'W',
                        xray_operating_V = 80, # in keV
                        xray_tilt_theta = 12, # in deg
                        xray_filter_material = 'Al',
                        xray_filter_thickness = 0.3, # in cm
                        imaging_samples = imaging_samples,
                        FoM_type = 'N_energy_bins', # K_edge_imaging
                        )
    
    simulation.cmake()
    simulation.make()
    
    fom = simulation.FoM(0, thickness=np.array([0.527273,0.809091,0.184700])) # in mm
    print("FoM: " + str(fom))
    
    # wvl_hist, wvl_bin = simulation.analyze_all_energies(0, thickness=np.array([0.527273,0.809091,0.184700]), pht_per_energy=simulation.Nphoton*np.ones(simulation.energy_bins))