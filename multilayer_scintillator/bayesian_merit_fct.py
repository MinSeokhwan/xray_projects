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
import fcntl

class geant4:
    def __init__(self, G4directory, dimScint, dimDet, gap, structureID, Nlayers, scintillator_material,
                 max_thickness, Ngrid, detector_thickness, check_overlap, energy_range, energy_bins,
                 Nphoton, wvl_bins, calibration_method):
        self.G4directory = G4directory
        
        self.dimScint = dimScint
        self.dimDet = dimDet
        self.dimWorld = np.max(np.vstack((dimScint, dimDet)), axis=0)
        self.gap = gap
        
        self.structureID = structureID
        self.Nlayers = Nlayers
        self.scintillator_material = scintillator_material
        
        self.max_thickness = max_thickness
        
        self.Ngrid = Ngrid
        self.detector_thickness = detector_thickness
        
        self.check_overlap = check_overlap
        
        # self.Nthreads = Nthreads
        
        self.energy_list = np.linspace(energy_range[0], energy_range[1], energy_bins)
        self.energy_bins = energy_bins
        self.Nphoton = Nphoton
        self.wvl_bins = wvl_bins
        
        self.calibration_method = calibration_method
        self.calibration_done = False
        
        # proc = subprocess.Popen([".", "/home/gridsan/smin/geant4/geant4-v11.1.2-install/share/Geant4/geant4make/geant4make.sh"])
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["cmake", ".."])
        time.sleep(2)
        proc = subprocess.Popen(["make", "-j4"])
        time.sleep(10)
        
        with open(directory + "/data/rgb_sensitivity.txt", 'r') as txt:
            line_count = 0
            for line in txt:
                line_count += 1
            txt.seek(0)
            self.sensRGB = np.zeros((line_count, 4))
            count = 0
            for line in txt:
                temp_array = np.array([line.split()])
                self.sensRGB[count,:] = np.asfarray(temp_array, float)
                count += 1
        
    def multicolor_scintillator_FoM(self, thickness):
        wvl_hist_all, wvl_bin_centers = self.test_all_energies(thickness)
        # wvl_hist_all, wvl_bin_centers = self.analyze_all_energies(thickness)
        # np.savez(directory + "/RGBU_debug" + str(int(self.permutationID)), wvl_hist_all=wvl_hist_all, wvl_bin_centers=wvl_bin_centers)
        # assert False
        
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
            if not self.calibration_done:
                self.reference_white_RGB = np.zeros((self.wvl_bins, 3))
                for i in range(3):
                    thickness_temp = 1e-6*np.ones(3)
                    thickness_temp[i] = self.max_thickness
                    wvl_hist_temp, _ = self.test_all_energies(thickness_temp, calibration=True)
                    self.reference_white_RGB[:,i] = wvl_hist_temp[:,-1]
                
                self.sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
                for i in range(3):
                    self.sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
                    self.sensRGB_interp[:,i] /= 100
                self.sensRGB_interp[:,[0,2]] /= 4
                self.sensRGB_interp[:,1] /= 2
                
                self.Nph_RGB = np.zeros(3)
                for i in range(3):
                    self.Nph_RGB[i] = np.dot(self.sensRGB_interp[:,i], self.reference_white_RGB[:,i])
                self.Nph_norm = np.min(self.Nph_RGB) #np.sum(self.Nph_RGB)/self.Nph_RGB.size
                
                self.calibration_done = True
            
            RGB_unclip = (self.sensRGB_interp.T @ wvl_hist_all)/self.Nph_norm
            RGB = RGB_unclip.copy()
            RGB[RGB > 1] = 1
            
            np.savez(directory + '/data/calibration_data', wvl_bin_centers=wvl_bin_centers,
                     sensRGB=self.sensRGB_interp, Nph_norm=self.Nph_norm, Nph_RGB=self.Nph_RGB,
                     reference_white_RGB=self.reference_white_RGB, RGB_unclip=RGB_unclip, RGB=RGB, wvl_hist_all=wvl_hist_all)
        
        # wvl_hist_all_norm = wvl_hist_all/np.sqrt(np.sum(wvl_hist_all**2, axis=0))[np.newaxis,:]
        avg_dist = 0
        for i in range(self.energy_bins):
            # avg_dist += np.sum(np.sqrt(np.sum((RGB[:,i][:,np.newaxis] - RGB)**2, axis=0)))
            avg_dist += np.sum(np.sqrt(np.sum((wvl_hist_all[:,i][:,np.newaxis] - wvl_hist_all)**2, axis=0)))
            # avg_dist -= np.sum(wvl_hist_all_norm[:,i][:,np.newaxis]*wvl_hist_all_norm[:,np.arange(self.energy_bins)!=i])
        avg_dist /= self.energy_bins*(self.energy_bins - 1)
        
        # RGBvec = RGB - 0.5
        # RGBvec /= np.sqrt(np.sum(RGBvec**2, axis=0))[np.newaxis,:]
        # s = svd(RGBvec, compute_uv=False)
        # (s**2 - 1)/(self.Nsample - 1)
        
        # RGBcov = np.cov(RGBvec)
        # np.trace(RGBcov)
        
        np.savez(directory + "/results/RGB_data_perm" + str(int(self.permutationID)), wvl_hist_all=wvl_hist_all, wvl_bin_centers=wvl_bin_centers,
                 RGB=RGB, avg_dist=avg_dist)
        
        return avg_dist
    
    def write_mac(self, thickness, source_energy):
        # for i in range(self.Nthreads):
        #     if os.path.exists(self.G4directory + "/buildMT/output_t" + str(int(i)) + ".root"):
        #         os.remove(self.G4directory + "/buildMT/output_t" + str(int(i)) + ".root")
        # if os.path.exists(self.G4directory + "/buildMT/output_total.root"):
        #     os.remove(self.G4directory + "/buildMT/output_total.root")
        if os.path.exists(self.G4directory + "/build/output.root"):
             os.remove(self.G4directory + "/build/output.root")
        
        with open(self.G4directory + "/mac/run_detector.mac", "w") as mac:
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
        
        with open(self.G4directory + "/mac/run_scintillator.mac", "w") as mac:
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
            
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["cmake", ".."])
        time.sleep(2)
    
    def read_output(self, n, detector, thickness):
        os.chdir(self.G4directory + "/build")
        if detector:
            proc = subprocess.Popen(["./NS", "run_detector.mac"])
        else:
            proc = subprocess.Popen(["./NS", "run_scintillator.mac"]) 
        
        # output_file_checks = np.zeros(self.Nthreads).astype(bool)
        # while True:
        #     for i in range(self.Nthreads):
        #         if os.path.exists(self.G4directory + "/buildMT/output_t" + str(int(i)) + ".root"):
        #             output_file_checks[i] = True
        #     if np.all(output_file_checks):
        #         break
        #     time.sleep(1)
        while True:
            if os.path.exists(self.G4directory + "/build/output.root"):
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
                root_file = uproot.open(self.G4directory + "/build/output.root")
                photons = root_file['Photons']
                print('')
                break
            except:
                print('| ', end='', flush=True)
                time.sleep(1)
        
        # os.chdir(self.G4directory + "/build")
        # hadd_list = ["hadd", "output_total.root"]
        # for i in range(self.Nthreads):
        #     hadd_list.append("output_t" + str(int(i)) + ".root")
        # proc = subprocess.Popen(hadd_list)
        # time.sleep(5)
        # root_file = uproot.open(self.G4directory + "/buildMT/output_total.root")
        
        # try:
        #     root_file = uproot.open(self.G4directory + "/build/output.root")
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
            
            np.savez(directory + "/results/analysis_data_energy" + str(int(n)) + "_perm" + str(int(self.permutationID)),
                     wvl_hist=wvl_hist, wvl_bin_centers=wvl_bin_centers,
                     spec_YAGCe=spec_YAGCe, spec_ZnSeTe=spec_ZnSeTe,
                     spec_LYSOCe=spec_LYSOCe, z_bin_edges=z_bin_edges,
                     z_YAGCe=z_YAGCe, z_ZnSeTe=z_ZnSeTe, z_LYSOCe=z_LYSOCe)
        
        return wvl_hist, wvl_bin_centers
    
    def test_all_energies(self, thickness, calibration=False):
        wvl_hist_all = np.zeros((self.wvl_bins, self.energy_bins))
        if calibration:
            n_start = self.energy_bins - 1
        else:
            n_start = 0
            
        for n in range(n_start, self.energy_bins):
            print('\n### Source Energy: ' + str(np.round(self.energy_list[n], 6)) + ' keV\n')
            t1 = time.time()
            self.run_geant4(thickness, self.energy_list[n])
            
            wvl_hist, wvl_bin_centers = self.read_output(n, detector=True, thickness=thickness)
            wvl_hist_all[:,n] = wvl_hist
            t2 = time.time()
            print('\n### Simulation Time: ' + str(t2 - t1) + ' s\n')
        
        return wvl_hist_all, wvl_bin_centers
    
    def analyze_all_energies(self, thickness):
        wvl_hist_all = np.zeros((self.wvl_bins, self.energy_bins))
        for n in range(self.energy_bins):
            print('\n### Source Energy: ' + str(np.round(self.energy_list[n], 6)) + ' keV\n')
            t1 = time.time()
            self.run_geant4(thickness, self.energy_list[n])
            
            wvl_hist, wvl_bin_centers = self.read_output(n, detector=False, thickness=thickness)
            wvl_hist_all[:,n] = wvl_hist
            t2 = time.time()
            print('\n### Simulation Time: ' + str(t2 - t1) + ' s\n')
        
        return wvl_hist_all, wvl_bin_centers

if __name__ == '__main__':
    simulation = geant4(G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main",
                        dimScint = np.array([1,1]), # in cm
                        dimDet = np.array([1,1]), # in cm
                        gap = 10, # in cm
                        structureID = 1,
                        Nlayers = 3,
                        scintillator_material = np.array([2,1,3]),
                        max_thickness = 1, # in mm
                        Ngrid = np.array([1,1]),
                        detector_thickness = 0.1, # in um
                        check_overlap = 1,
                        # Nthreads = 8,
                        energy_range = np.array([10,100]), # in keV
                        energy_bins = 10,
                        Nphoton = 1000000,
                        wvl_bins = 801,
                        calibration_method = 'photon_count')
    simulation.permutationID = 22
    # fom = simulation.multicolor_scintillator_FoM(thickness=np.array([0.939103,0.945647,0.259164])) # in mm
    fom = simulation.multicolor_scintillator_FoM(thickness=np.array([0.581986,0.903233,0.218789])) # in mm
    print("FoM: " + str(fom))