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
                 rSource,
                 sampleID,
                 dimSampleVoxel,
                 indSampleVoxel,
                 IConc,
                 GdConc,
                 dimScintCorr,
                 dimScintImg,
                 matScintCorr,
                 matScintImg,
                 dimDetVoxel,
                 NDetVoxel,
                 gapSampleScint,
                 gapScintDet,
                 check_overlap,
                 energy_range,
                 energy_bins,
                 theta_bins,
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

        # Source
        self.rSource = rSource
        self.theta_list = np.linspace(0, 2*np.pi, theta_bins, endpoint=False)
        self.energy_list = np.linspace(energy_range[0], energy_range[1], energy_bins)
        self.energy_bins = energy_bins
        self.energy_list, xraySrc = xray.get_spectrum(xray_target_material,
                                                      self.energy_list,
                                                      xray_operating_V,
                                                      xray_tilt_theta,
                                                      xray_filter_material,
                                                      xray_filter_thickness,
                                                      include_char_xray=False,
                                                      )
        self.energy_list = np.delete(self.energy_list, np.where(xraySrc==0)[0])
        xraySrc = np.delete(xraySrc, np.where(xraySrc==0)[0])
        
        self.energy_bins = self.energy_list.size
        self.xraySrc = xraySrc/np.sum(xraySrc)
        
        # Sample
        self.sampleID = sampleID
        self.dimSampleVoxel = dimSampleVoxel
        self.indSampleVoxel = indSampleVoxel
        self.NSampleVoxel = indSampleVoxel[:,1] - indSampleVoxel[:,0]
        self.dimSample = self.NSampleVoxel*dimSampleVoxel # cm
        self.IConc = IConc
        self.GdConc = GdConc

        # Scintillator
        self.dimScintCorr = dimScintCorr
        self.dimScintImg = dimScintImg
        self.matScintCorr = matScintCorr
        self.matScintImg = matScintImg
        
        # Detector
        self.dimDetVoxel = dimDetVoxel
        self.NDetVoxel = NDetVoxel
        self.dimDet = NDetVoxel*dimDetVoxel # um
        
        # World
        self.dimWorld = np.zeros(3)
        self.dimWorld[0] = np.max((self.dimScintCorr[1], self.dimSample[0], self.dimDet[0]))
        self.dimWorld[1] = rSource
        self.dimWorld[2] = rSource

        self.gapSampleScint = gapSampleScint
        self.gapScintDet = gapScintDet
        
        # Misc.
        self.check_overlap = check_overlap
        self.verbose = verbose
        self.identifier = identifier
        self.rank_start = rank_start
    
    def make_simulation_macro(self, Nruns, souce_theta=0.0, source_type='Point'):
        with open(self.G4directory + "/build/run_detector_rank" + str(comm.rank + self.rank_start) + ".mac", "w") as mac:
            mac.write("/system/root_file_name output" + str(comm.rank + self.rank_start) + ".root\n")
        
            mac.write("/structure/rWorld " + str(np.round(self.dimWorld[0], 6)) + "\n")
            mac.write("/structure/zWorld " + str(np.round(self.dimWorld[1], 6)) + "\n\n")

            mac.write("/structure/gapSampleScint " + str(np.round(self.gapSampleScint, 6)) + "\n")
            mac.write("/structure/gapScintDet " + str(np.round(self.gapScintDet, 6)) + "\n\n")
            
            mac.write("/structure/rScintCorr " + str(np.round(self.dimScintCorr[0], 6)) + "\n")
            mac.write("/structure/zScintCorr " + str(np.round(self.dimScintCorr[1], 6)) + "\n")
            mac.write("/structure/xScintImg " + str(np.round(self.dimScintImg[0], 6)) + "\n")
            mac.write("/structure/yScintImg " + str(np.round(self.dimScintImg[1], 6)) + "\n")
            mac.write("/structure/zScintImg " + str(np.round(self.dimScintImg[2], 6)) + "\n")
            mac.write("/structure/matScintCorr " + str(self.matScintCorr) + "\n")
            mac.write("/structure/matScintImg " + str(self.matScintImg) + "\n\n")
            
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
            
            mac.write("/structure/xDet " + str(np.round(self.dimDet[0], 6)) + "\n")
            mac.write("/structure/yDet " + str(np.round(self.dimDet[1], 6)) + "\n")
            mac.write("/structure/nDetX " + str(int(self.NDetVoxel[0])) + "\n")
            mac.write("/structure/nDetY " + str(int(self.NDetVoxel[1])) + "\n")
            mac.write("/structure/detectorDepth " + str(np.round(self.dimDet[2], 6)) + "\n\n")
            
            mac.write("/structure/checkDetectorsOverlaps " + str(int(self.check_overlap)) + "\n\n")
            
            mac.write("/run/numberOfThreads 1\n")
            mac.write("/run/initialize\n")
            
            mac.write("/gps/particle gamma\n")
            mac.write("/gps/direction "\
            	+ "0 "\
                + str(np.round(np.cos(source_theta - np.pi/2), 6)) + " "\
                + str(np.round(np.sin(source_theta - np.pi/2), 6)) + "\n") # downward incidence is theta = 0
            mac.write("/gps/pos/type " + source_type + "\n")
            mac.write("/gps/pos/centre"\
            	+ " 0.0 "\
                + " " + str(np.round(self.dimWorld[0]*np.cos(source_theta + np.pi/2), 6))\
                + " " + str(np.round(self.dimWorld[0]*np.sin(source_theta + np.pi/2), 6)) + "\n")
            mac.write("/gps/ene/type User\n")
            mac.write("/gps/ene/emspec 0\n")
            mac.write("/gps/hist/type energy\n")
            mac.write("/gps/hist/point 0. 0.\n")
            for ne in range(self.energy_bins):
                mac.write("/gps/hist/point " + str(np.round(self.energy_list[ne]/1e3, 6)) + " " + str(np.round(self.xraySrc[ne], 6)) + "\n")

            mac.write("/run/beamOn " + str(int(Nruns)))
        
    def run_geant4(self, Nruns, source_theta):
        if os.path.exists(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root"):
            os.remove(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root")
    
        os.chdir(self.G4directory + "/build")
        time.sleep(comm.size*np.random.rand())
        proc = subprocess.Popen(["./NS", "run_detector_rank" + str(comm.rank + self.rank_start) + ".mac"])
        
        while True:
            if os.path.exists(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root"):
                break
            time.sleep(1)
        
        while True:
            try:
                root_file = uproot.open(self.G4directory + "/build/output" + str(comm.rank + self.rank_start) + "_t0.root")
                photons = root_file['Photons']
                hits = root_file['Hits']
                source = root_file['Source']
                process = root_file['Process']
                break
            except:
                time.sleep(1)

        eventID = np.array(photons['fEvent'].array())
        wvl_list = np.array(photons['fWlen'].array())
        particle_type_list = np.array(photons['particleType'].array())
        fz_list = np.array(hits["fZ"].array())
        E_inc_list = np.array(source['fEnergy'].array())
        n_phot_list = np.array(process['Photoelectric'].array())
        n_compt_list = np.array(process['Compton'].array())
        
        # Database Quantities
        theta_inc = np.zeros(())
        E_X_det = np.zeros((self.theta_bins, ))

        if os.path.exists(directory + "/data/" + self.identifier + "/geant4_data_" + str(comm.rank + self.rank_start) + ".npz"):
            with np.load(directory + "/data/" + self.identifier + "/geant4_data_" + str(comm.rank + self.rank_start) + ".npz") as data:
                E_inc = data['E_inc']
                theta_inc = data['theta_inc']
                E_X_corr = data['E_X_corr']
                E_X_det = data['E_X_det']
                n_scint_corr = data['n_scint_corr']
                n_scint_det = data['n_scint_det']
                n_phot = data['n_phot']
                n_compt = data['n_compt']
        
                E_inc = np.hstack((E_inc_list, E_inc))
                theta_inc = np.hstack((source_theta*np.ones(Nruns), theta_inc))
                E_X_corr = np.hstack((np.zeros(Nruns), E_X_corr))
                E_X_det = np.hstack((np.zeros(Nruns), E_X_det))
                n_scint_corr = np.hstack((np.zeros(Nruns), n_scint_corr))
                n_scint_det = np.hstack((np.zeros(Nruns), n_scint_det))
                n_phot = np.hstack((n_phot_list, n_phot))
                n_compt = np.hstack((n_compt_list, n_compt))
                
        else:
            E_inc = E_inc_list.copy() # Incident X-ray energy (keV)
            theta_inc = source_theta*np.ones(Nruns) # Incidence Angle (rad)
            E_X_corr = np.zeros(Nruns) # X-ray energy measured by the correlation detector (keV)
            E_X_det = np.zeros(Nruns) # X-ray energy measured by the imaging detector (keV)
            n_scint_corr = np.zeros(Nruns) # Scintillation intensity on the correlation detector (photons)
            n_scint_det = np.zeros(Nruns) # Scintillation intensity on the imaging detector (photons)
            n_phot = n_phot_list.copy() # Number of photoelectric absorption events
            n_compt = n_compt_list.copy() # Number of Compton scattering events
        
        z_upper = np.max(fz_list)
        z_lower = np.min(fz_list)

        for i in range(Nruns):
            event_mask = eventID == i
            
            if event_mask.size != 0:
                particleType_i = particleType[event_mask]
                z_upper_i = fz_list[event_mask] == z_upper
                
                if np.any(particleType_i == 'gamma'):
                    ind_gamma = np.argwhere(particleType_i == 'gamma')[:,0]
                
                    # Detected X-Ray Energy
                    for ig in ind_gamma:
                        if fz_list[event_mask][ig] == z_upper:
                            E_X_corr[i] = 1.239841939/wvl_list[event_mask][ig]
                        elif fz_list[event_mask][ig] == z_lower:
                            E_X_det[i] = 1.239841939/wvl_list[event_mask][ig]
                        else:
                            print("fz = " + str(fz_list[event_mask][ig]) + " not z_upper = " + str(z_upper) + " or z_lower = " + str(z_lower), flush=True)
                            assert False

                else:
                    E_X_corr[i] = np.nan
                    E_X_det[i] = np.nan
            
                if np.any(particleType_i == 'opticalphoton'):
                    mask_opticalphoton = particleType_i == 'opticalphoton'

                    # Total number of scintillation photons
                    n_scint[i] = np.sum(mask_opticalphoton)
                
                else:
                    n_scint[i] = 0
                
        np.savez(directory + "/data/" + self.identifier + "/geant4_data_" + str(comm.rank + self.rank_start),
                 E_inc=E_inc,
                 theta_inc=theta_inc,
                 E_X_corr=E_X_corr,
                 E_X_det=E_X_det,
                 n_scint_corr=n_scint_corr,
                 n_scint_det=n_scint_det,
                 n_phot=n_phot,
                 n_compt=n_compt,
                 )
    
    def collect_simulation_data(self, Nphot_per_theta):
        if verbose:
            print('\n### Collecting Simulation Data', flush=True)
            print('    ', end='', flush=True)
        
        m = self.theta_bins // size
        r = self.theta_bins % size
        theta_start = rank*m
        theta_end = (rank + 1)*m + r*(rank == size - 1)
        for nT in range(theta_start, theta_end):
            if verbose:
                if nT%10 == 0:
                    print(nT, end='', flush=True)
                else:
                    print('/', end='', flush=True)
            
            self.make_simulation_macro(Nphot_per_theta, source_theta=self.theta_list[nT], source_type='Point')
            self.read_output(Nphot_per_theta, source_theta=self.theta_list[nT])
        
        if verbose:
            print(i+1, flush=True)

if __name__ == '__main__':
    if rank == 0:
        verbose = True
    else:
        verbose = False
    
    identifier = 'YAGCe_50_120keV_tube_Pb3mm'
    
    if rank == 0:
        if not os.path.exists(directory + "/data/" + identifier):
            os.mkdir(directory + "/data/" + identifier)

    # Scintillator Index: 1:YAGCe, 2:ZnSeTe, 3:LYSOCe, 4:CsITl, 5:GSOCe, 6:NaITl, 7:GadoxTb, 8:YAGYb

    simulation = geant4(
        G4directory = "/nfs/scistore08/roquegrp/smin/xray_projects/geant4/G4_compton_correlation",
        rSource = 50.0, # in cm
        sampleID = 0, # 0: None, 1:AM, 2:AF, 3:Cylindrical
        dimSampleVoxel = np.array([2.137,8,0.0])/np.array([400,400,1]), # in cm
        indSampleVoxel = np.array([[0,253], # 0...253 indexing
                                    [119,189], # 0...222 indexing
                                    [63,64]]), # 1...128 indexing
        IConc = 0.01,
        GdConc = 0.01,
        dimScintCorr = np.array([0.5,0.1]), # [r,z] in cm
        dimScintImg = np.array([0.5,0.5,0.1]), # [x,y,z] in cm
        matScintCorr = 1, # YAGCe
        macScintImg = 3, # LYSOCe
        dimDetVoxel = np.array([100000.,100000.,0.1]), # in um
        NDetVoxel = np.array([1,1,1]),
        gapSampleScint = 0.001, # in cm
        gapScintDet = 0.001, # in cm
        check_overlap = 0,
        energy_range = np.array([50,120]), # in keV
        energy_bins = 71,
        theta_bins = 360,
        xray_target_material = 'W',
        xray_operating_V = 120, # in keV
        xray_tilt_theta = 12, # in deg
        xray_filter_material = ['Pb'],
        xray_filter_thickness = [0.3], # in cm
        verbose = verbose,
        identifier = identifier, #-------------------------------------------------------- Change RGB_150keV_nogap_phantom
        rank_start = 0, #----------------------------------------------------------------- Change
    )

    simulation.collect_simulation_data(Nreps=100, Nphot_per_rep=15000)
