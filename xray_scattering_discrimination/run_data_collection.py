import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-31])

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt 
from itertools import product
import shutil
import subprocess
import uproot
import time
import util.xray_spectrum_generator as xray

from mpi4py import MPI
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

class geant4:
    def __init__(self,
                 G4directory,
                 dimScint,
                 dimDetVoxel,
                 NDetVoxel,
                 gapScintDet,
                 Nlayers,
                 scintillator_material,
                 scintillator_thickness,
                 check_overlap,
                 energy_range,
                 energy_bins,
                 xray_target_material=None,
                 xray_operating_V=None,
                 xray_tilt_theta=None,
                 xray_filter_material=None,
                 xray_filter_thickness=None,
                 verbose=False,
                 identifier='',
                 rank_start=0,
                 ):
                 
        self.G4directory = G4directory
        
        # Scintillator
        self.dimScint = dimScint # cm
        self.Nlayers = Nlayers
        self.scintillator_material = scintillator_material
        self.scintillator_thickness = scintillator_thickness
        
        # Detector
        self.dimDetVoxel = dimDetVoxel
        self.NDetVoxel = NDetVoxel
        self.dimDet = NDetVoxel*dimDetVoxel # um
        
        # World
        self.dimWorld = np.max(np.vstack((dimScint, self.dimDet[:2]/1e4)), axis=0) # cm
        self.gapSampleScint = 0
        self.gapScintDet = gapScintDet
        self.dzWorld = 1.01*(self.dimDet[2]/1e4 + gapScintDet + np.sum(self.scintillator_thickness)/10 + gapScintDet + self.dimDet[2]/1e4) # cm
        
        self.check_overlap = check_overlap
        
        self.energy_list = np.linspace(energy_range[0], energy_range[1], energy_bins)
        self.energy_bins = energy_bins
        
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
        self.identifier = identifier
        self.rank_start = rank_start
    
    def make_simulation_macro(self, Nruns, source_energy=None, source_xy=None):
        with open(self.G4directory + "/build/run_detector_rank" + str(comm.rank + self.rank_start) + ".mac", "w") as mac:
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
            
            mac.write("/structure/angleFilter false\n")
            mac.write("\n")
            
            mac.write("/structure/sampleID 0\n")
            mac.write("/structure/xSample 0\n")
            mac.write("/structure/ySample 0\n")
            mac.write("/structure/zSample 0\n")
            mac.write("/structure/nSampleX 0\n")
            mac.write("/structure/nSampleY 0\n")
            mac.write("/structure/nSampleZ 0\n")
            mac.write("/structure/indSampleX 0\n")
            mac.write("/structure/indSampleY 0\n")
            mac.write("/structure/indSampleZ 0\n")
            mac.write("/structure/IConcentration 0\n")
            mac.write("/structure/GdConcentration 0\n")
            mac.write("\n")
            
            mac.write("/structure/constructDetectors 1\n")
            mac.write("/structure/constructTopDetector 1\n")
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
        if os.path.exists(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root"):
            os.remove(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root")
    
        os.chdir(self.G4directory + "/build")
        proc = subprocess.Popen(["./NS", "run_detector_rank" + str(comm.rank + self.rank_start) + ".mac"])
        
        while True:
            if os.path.exists(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root"):
                break
            time.sleep(1)
        
        while True:
            try:
                root_file = uproot.open(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root")
                photons = root_file['Photons']
                source = root_file['Source']
                process = root_file['Process']
                break
            except:
                time.sleep(1)

        eventID = np.array(photons['fEvent'].array())
        particleType = np.array(photons['particleType'].array())
        wvl_list = np.array(photons["fWlen"].array())
        fx_list = np.array(photons["fX"].array())
        fy_list = np.array(photons["fY"].array())
        fxy_list = np.vstack((fx_list, fy_list)).T
        fz_list = np.array(photons["fZ"].array())
        px_list = np.array(photons["pX"].array())
        py_list = np.array(photons["pY"].array())
        pz_list = np.array(photons["pZ"].array())
        vx_list = np.array(photons["vX"].array())
        vy_list = np.array(photons["vY"].array())
        vz_list = np.array(photons["vZ"].array())
        
        # Database Quantities
        if os.path.exists(directory + "/data/" + self.identifier + "/geant4_data_" + str(comm.rank + self.rank_start) + ".npz"):
            with np.load(directory + "/data/" + self.identifier + "/geant4_data_" + str(comm.rank + self.rank_start) + ".npz") as data:
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
        
                E_inc = np.hstack((np.array(source["fEnergy"].array())*1e3, E_inc))
                dE_trans = np.hstack((np.zeros(Nruns), dE_trans))
                theta_trans = np.hstack((np.zeros(Nruns), theta_trans))
                depth_scint = np.hstack((np.zeros(Nruns), depth_scint))
                R_scint = np.vstack((np.zeros((Nruns, 2)), R_scint))
                dxy_scint = np.vstack((np.zeros((Nruns, 2)), dxy_scint))
                n_scint = np.hstack((np.zeros(Nruns), n_scint))
                n_rayl = np.hstack((np.array(process["Rayleigh"].array()), n_rayl))
                n_phot = np.hstack((np.array(process["Photoelectric"].array()), n_phot))
                n_compt = np.hstack((np.array(process["Compton"].array()), n_compt))
                
        else:
            E_inc = np.array(source["fEnergy"].array())*1e3 # Incident Energy (keV)
            dE_trans = np.zeros(Nruns) # Energy Loss of Transmitted X-Ray (keV)
            theta_trans = np.zeros(Nruns) # X-Ray Scattering Angle (deg)
            depth_scint = np.zeros(Nruns) # Average scintillation depth (mm)
            R_scint = np.zeros((Nruns, 2)) # Scintillation spot size (top & bottom; mm)
            dxy_scint = np.zeros((Nruns, 2)) # Scintillation spot displacement (top & bottom; mm)
            n_scint = np.zeros(Nruns) # Total number of scintillation photons
            n_rayl = np.array(process["Rayleigh"].array()) # Number of Rayleigh scattering events
            n_phot = np.array(process["Photoelectric"].array()) # Number of photoelectric effect events
            n_compt = np.array(process["Compton"].array()) # Number of Compton scattering events
        
        if rank == 0:
            np.savez(directory + "/debug_particleType", eventID=eventID, particleType=particleType)
        
        for i in range(Nruns):
            event_mask = eventID == i
            
            if event_mask.size != 0:
                particleType_i = particleType[event_mask]
                
                if np.any(particleType_i == 'gamma'):
                    ind_gamma = np.argwhere(particleType_i == 'gamma')
                
                    # Energy Loss of Transmitted X-Ray
                    E_trans = 1.239841939/wvl_list[event_mask][ind_gamma]
                    dE_trans[i] = E_inc[i] - E_trans
                    
                    # X-Ray Scattering Angle
                    pxy = np.sqrt(px_list[event_mask][ind_gamma]**2 + py_list[event_mask][ind_gamma]**2)
                    theta_trans[i] = np.arctan2(pxy, pz_list[event_mask][ind_gamma])

                else:
                    dE_trans[i] = E_inc[i]
                    theta_trans[i] = np.nan
            
                if np.any(particleType_i == 'opticalphoton'):
                    mask_opticalphoton = particleType_i == 'opticalphoton'

                    # Average scintillation depth
                    z_upper = 10*(-self.dzWorld/2 + self.dimDet[2]/1e4 + self.gapScintDet) # mm
                    z_scint = np.mean(vz_list[event_mask][mask_opticalphoton])
                    depth_scint[i] = z_scint - z_upper
                    
                    # Scintillation spot size & displacement
                    fz_i_opt = fz_list[event_mask][mask_opticalphoton]
                    mask_upperDetector = fz_i_opt < 0
                    mask_lowerDetector = fz_i_opt > 0
                    
                    fxy_i_opt = fxy_list[event_mask,:][mask_opticalphoton,:]
                    
                    centroid = np.nan*np.ones((2, 2))
                    if mask_upperDetector.size > 0:
                        centroid[:,0] = np.mean(fxy_i_opt[mask_upperDetector,:], axis=0)
                    if mask_lowerDetector.size > 0:
                        centroid[:,1] = np.mean(fxy_i_opt[mask_lowerDetector,:], axis=0)

                    R_scint[i,0] = np.mean(np.sqrt(np.sum((centroid[:,0][np.newaxis,:] - fxy_i_opt[mask_upperDetector,:])**2, axis=1)))
                    R_scint[i,1] = np.mean(np.sqrt(np.sum((centroid[:,1][np.newaxis,:] - fxy_i_opt[mask_lowerDetector,:])**2, axis=1)))
                    
                    dxy_scint[i,:] = np.sqrt(np.sum(centroid**2, axis=0))
                
                    # Total number of scintillation photons
                    n_scint[i] = np.sum(mask_opticalphoton)
                
                else:
                    depth_scint[i] = np.nan
                    R_scint[i,:] = np.nan
                    dxy_scint[i,:] = np.nan
                    n_scint[i] = 0
                
        np.savez(directory + "/data/" + self.identifier + "/geant4_data_" + str(comm.rank + self.rank_start),
                 E_inc=E_inc,
                 dE_trans=dE_trans,
                 theta_trans=theta_trans,
                 depth_scint=depth_scint,
                 R_scint=R_scint,
                 dxy_scint=dxy_scint,
                 n_scint=n_scint,
                 n_rayl=n_rayl,
                 n_phot=n_phot,
                 n_compt=n_compt,
                 )
    
    def collect_simulation_data(self, Nreps, Nruns_per_energy):
        if verbose:
            print('\n### Collecting Simulation Data', flush=True)
            print('    ', end='', flush=True)
            
        for i in range(Nreps):
            if verbose:
                if i%10 == 0:
                    print(i, end='', flush=True)
                else:
                    print('/', end='', flush=True)
            
            for nE in range(self.energy_bins):
                self.make_simulation_macro(Nruns_per_energy, source_energy=self.energy_list[nE], source_xy='center')
                self.read_output(Nruns_per_energy)
        
        if verbose:
            print(i+1, flush=True)

if __name__ == '__main__':
    if rank == 0:
        verbose = True
    else:
        verbose = False
    
    identifier = 'YAGCe_150keV'
    
    if rank == 0:
        if not os.path.exists(directory + "/data/" + identifier):
            os.mkdir(directory + "/data/" + identifier)

    simulation = geant4(G4directory = "/home/minseokhwan/xray_projects/geant4/G4_Nanophotonic_Scintillator-main",
                        dimScint = np.array([100.,100.]), # in cm
                        dimDetVoxel = np.array([1000000.,1000000.,0.1]), # in um
                        NDetVoxel = np.array([1,1,1]),
                        gapScintDet = 0.001, # in cm
                        Nlayers = 1, #---------------------------------------------------------------------- Change
                        scintillator_material = np.array([1]), # 1:YAGCe, 2:ZnSeTe, 3:LYSOCe, 4:CsITl, 5:GSOCe, 6:NaITl, 7:GadoxTb, 8:YAGYb #- Change
                        scintillator_thickness = np.array([10.]), # in mm #---------- Change 0.222604,0.313417,0.337213 | 0.076081,0.011322,0.073472
                        check_overlap = 0,
                        energy_range = np.array([10,150]), # in keV
                        energy_bins = 141,
                        xray_target_material = 'W',
                        xray_operating_V = 150, # in keV
                        xray_tilt_theta = 12, # in deg
                        xray_filter_material = ['Al'],
                        xray_filter_thickness = [0.3], # in cm
                        verbose = verbose,
                        identifier = identifier, #-------------------------------------------------------- Change RGB_150keV_nogap_phantom
                        rank_start = 0, #----------------------------------------------------------------- Change
                        )
    
    simulation.collect_simulation_data(Nreps=100, Nruns_per_energy=42)