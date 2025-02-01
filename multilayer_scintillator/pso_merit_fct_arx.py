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
#            dE_ref = (self.energy_list[1] - self.energy_list[0])/2
#            self.xray_scale = np.zeros(self.energy_bins)
#            for ne in range(self.energy_bins):
#                if ne == 0:
#                    self.xray_scale[ne] = (self.energy_list[ne+1] - self.energy_list[ne])/dE_ref
#                elif ne == self.energy_bins - 1:
#                    self.xray_scale[ne] = (self.energy_list[ne] - self.energy_list[ne-1])/dE_ref
#                else:
#                    self.xray_scale[ne] = (self.energy_list[ne+1]/2 - self.energy_list[ne-1]/2)/dE_ref
#            self.xraySrc *= self.xray_scale

        elif self.source_type == 'uniform':
            self.xraySrc = np.ones(self.energy_bins)/self.energy_bins
            
        if FoM_type == 'K_edge_imaging' or FoM_type == 'K_edge_imaging_v2':
#            self.Nsample = imaging_samples.size
#            self.linAttCoeff = np.zeros((self.energy_bins, self.Nsample))
#            for ns in range(self.Nsample):
#                raw = txt.read_txt(directory + '/data/linAttCoeff' + imaging_samples[ns])
#                self.linAttCoeff[:,ns] = np.interp(self.energy_list, raw[:,0], raw[:,1], left=0, right=0)

#            self.training_data = training_data
            
            _, self.linAttCoeff, self.training_data, self.sampled_data, _, self.linAttCoeff_svd, self.scale_thickness = gen.generate('AM', self.energy_list, 200)
            self.Nsample = self.linAttCoeff.shape[1]
            self.Ntrain = self.training_data.shape[0]
            _, self.Nx_img, self.Nz_img = self.sampled_data.shape
            
        elif FoM_type == 'K_edge_imaging_v3':
            self.linAttCoeff, _, _, _, self.train_thickness, _, _ = gen.generate('AM', self.energy_list, 200)
            _, _, _, _, self.test_thickness, _, _ = gen.generate('AF', self.energy_list, 200)
            
            self.Nsample = self.linAttCoeff.shape[1]
            _, self.Nx_img_train, self.Nz_img_train = self.train_thickness.shape
            _, self.Nx_img_test, self.Nz_img_test = self.test_thickness.shape
        
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
    
    def FoM(self, rank, thickness, gamma_optimization=False):
        if gamma_optimization:
            if comm.rank == 0:
                wvl_hist_temp, wvl_bin_centers = self.RGB_calibration(rank, thickness)
                wvl_hist_temp = wvl_hist_temp.reshape(-1)
            else:
                wvl_hist_temp = np.zeros(self.wvl_bins*self.energy_bins, dtype=np.float64)
                wvl_bin_centers = np.zeros(self.wvl_bins, dtype=np.float64)
                self.raw_clip = 0
            
            comm.Bcast(wvl_hist_temp, root=0)
            wvl_hist_temp = wvl_hist_temp.reshape(self.wvl_bins, self.energy_bins)
            comm.Bcast(wvl_bin_centers, root=0)
            self.raw_clip = comm.bcast(self.raw_clip, root=0)
        else:
            wvl_hist_temp, wvl_bin_centers = self.RGB_calibration(rank, thickness)
        
#        data = np.load(directory + '/results/RGB_data.npz')
#        wvl_hist_temp = data['wvl_hist_temp']
#        wvl_bin_centers = data['wvl_bin_centers']
#        sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
#        for i in range(3):
#            sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
#            sensRGB_interp[:,i] /= 100
#        self.raw_clip = np.max(sensRGB_interp.T @ wvl_hist_temp @ self.xraySrc)
    
        if self.FoM_type == 'N_energy_bins':
            # wvl_hist_all, wvl_bin_centers = self.test_all_energies(rank, thickness)
            # wvl_hist_all, wvl_bin_centers = self.analyze_all_energies(thickness)
            wvl_hist_all = wvl_hist_temp @ np.diag(self.xraySrc)
            
#            if np.max(wvl_hist_all) != 0:
#                wvl_hist_all /= np.max(wvl_hist_all)
            
            RGB, sensRGB_interp = self.RGB_compute(wvl_hist_all, wvl_bin_centers)
            
            RGB_tgt = np.zeros((3, 3))
            weight = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    if i == j:
                        RGB_tgt[i,j] = 1
                        weight[i,j] = 1
                    else:
                        RGB_tgt[i,j] = 0
                        weight[i,j] = 2
                        
            fom = np.sum(weight*(RGB - RGB_tgt)**2)
            
#            fom = 0
#            for i in range(wvl_hist_all.shape[1]):
#                fom += np.sum(np.sqrt(np.sum((wvl_hist_all[:,i][:,np.newaxis] - wvl_hist_all)**2, axis=0)))
#            fom /= wvl_hist_all.shape[1]*(wvl_hist_all.shape[1] - 1)
            
        elif self.FoM_type == 'K_edge_imaging':
            RGB_train, wvl_hist_all, xraySrc_fwd = self.ground_truth_forward_model(self.training_data, wvl_hist_temp, wvl_bin_centers) #/self.scale_thickness[np.newaxis,:]
            RGB_image, _, _ = self.ground_truth_forward_model(self.sampled_data.reshape(self.Nsample,-1).T, wvl_hist_temp, wvl_bin_centers)
            
            np.random.seed(0)
            RGB_noise = (RGB_train.T + 0.1*np.mean(RGB_train)*np.random.normal(size=(self.Ntrain,3))).reshape(-1)
            RGB_image_noise = (RGB_image.T + 0.1*np.mean(RGB_image)*np.random.normal(size=(self.Nx_img*self.Nz_img,3))).reshape(-1)
            np.random.seed()
            
            sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
            for i in range(3):
                sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
                sensRGB_interp[:,i] /= 100
            
            Di = sensRGB_interp.T @ wvl_hist_temp
            self.scale_intensity = 0.1*np.sqrt(np.sum(Di**2, axis=0))
#            wvl_hist_temp /= self.scale_intensity[np.newaxis,:]
#            xraySrc_fwd /= self.scale_intensity[:,np.newaxis]
            
            if gamma_optimization:
                t1 = time.time()
                if self.verbose:
                    print('\n### Step 1 Regularization Matrix Optimization', flush=True)
                self.gamma_block1_opt = self.regularization_matrix_optimization1(RGB_noise,
                                                                                 sensRGB_interp,
                                                                                 wvl_hist_temp,
                                                                                 xraySrc_fwd)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                t1 = time.time()
                if self.verbose:
                    print('\n### Transmitted X-Ray Spectrum Reconstruction (training data)', end='', flush=True)
                I_reconstr, _, _ = jit_fct.transmitted_intensity_reconstruction(RGB_noise,
                                                                                sensRGB_interp,
                                                                                wvl_hist_temp,
                                                                                self.gamma_block1_opt,
                                                                                self.energy_bins,
                                                                                self.Ntrain,
                                                                                self.raw_clip)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                    
                t1 = time.time()
                if self.verbose:
                    print('\n### Transmitted X-Ray Spectrum Reconstruction (sampled image)', end='', flush=True)
                I_image_reconstr, _, _ = jit_fct.transmitted_intensity_reconstruction(RGB_image_noise,
                                                                                      sensRGB_interp,
                                                                                      wvl_hist_temp,
                                                                                      self.gamma_block1_opt,
                                                                                      self.energy_bins,
                                                                                      self.Nx_img*self.Nz_img,
                                                                                      self.raw_clip)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                xraySrc_tile = np.tile(self.xraySrc, self.Ntrain)
                T_reconstr = I_reconstr/xraySrc_tile
                T_reconstr[T_reconstr<0] = np.min(T_reconstr[T_reconstr>0])
                
                xraySrc_tile = np.tile(self.xraySrc, self.Nx_img*self.Nz_img)
                T_image_reconstr = I_image_reconstr/xraySrc_tile
                T_image_reconstr[T_image_reconstr<0] = np.min(T_image_reconstr[T_image_reconstr>0])
                
                t1 = time.time()
                if self.verbose:
                    print('\n### Step 2 Regularization Matrix Optimization', flush=True)
                self.gamma_block2_opt = self.regularization_matrix_optimization2(T_reconstr)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                t1 = time.time()
                if self.verbose:
                    print('\n### Element Thickness Reconstruction (training data)', end='', flush=True)
                d_reconstr, _, _ = jit_fct.element_thickness_reconstruction(T_reconstr,
                                                                            self.linAttCoeff,
                                                                            self.gamma_block2_opt,
                                                                            self.energy_bins,
                                                                            self.Nsample,
                                                                            self.Ntrain)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                t1 = time.time()
                if self.verbose:
                    print('\n### Element Thickness Reconstruction (sampled image)', end='', flush=True)
                d_image_reconstr, _, _ = jit_fct.element_thickness_reconstruction(T_image_reconstr,
                                                                                  self.linAttCoeff,
                                                                                  self.gamma_block2_opt,
                                                                                  self.energy_bins,
                                                                                  self.Nsample,
                                                                                  self.Nx_img*self.Nz_img)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                # fom = np.sum(((d_reconstr.reshape(self.Ntrain, self.Nsample)[:,-1] - self.training_data[:,-1])**2)/training_data_magnitude)
                weights = 1/self.training_data.reshape(-1)
                weights[self.training_data.reshape(-1)==0] = np.max(weights[weights!=np.inf])
                weights[(np.arange(weights.size)%self.Nsample)!=self.Nsample-1] = 0
                fom = np.sum((d_reconstr - self.training_data.reshape(-1))**2)
            
            else:
                t1 = time.time()
                if self.verbose:
                    print('\n### Step 1 Regularization Matrix Optimization', flush=True)
                self.gamma_block1_opt = self.regularization_matrix_optimization1(RGB_noise,
                                                                                 sensRGB_interp,
                                                                                 wvl_hist_temp,
                                                                                 xraySrc_fwd)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                t1 = time.time()
                if self.verbose:
                    print('\n### Transmitted X-Ray Spectrum Reconstruction (training data)', end='', flush=True)
                I_reconstr, _, _ = jit_fct.transmitted_intensity_reconstruction(RGB_noise,
                                                                                sensRGB_interp,
                                                                                wvl_hist_temp,
                                                                                self.gamma_block1_opt,
                                                                                self.energy_bins,
                                                                                self.Ntrain,
                                                                                self.raw_clip)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                    
                t1 = time.time()
                if self.verbose:
                    print('\n### Transmitted X-Ray Spectrum Reconstruction (sampled image)', end='', flush=True)
                I_image_reconstr, _, _ = jit_fct.transmitted_intensity_reconstruction(RGB_image_noise,
                                                                                      sensRGB_interp,
                                                                                      wvl_hist_temp,
                                                                                      self.gamma_block1_opt,
                                                                                      self.energy_bins,
                                                                                      self.Nx_img*self.Nz_img,
                                                                                      self.raw_clip)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                weights = np.tile(1/np.sum(xraySrc_fwd.T**2, axis=1), (self.energy_bins,1)).T.reshape(-1)
                fom = np.sum(weights*(I_reconstr - xraySrc_fwd.T.reshape(-1))**2)
        
        elif self.FoM_type == 'K_edge_imaging_v2':
            RGB_train, wvl_hist_all, xraySrc_fwd = self.ground_truth_forward_model(self.training_data, wvl_hist_temp, wvl_bin_centers)
            
            np.random.seed(0)
            RGB_noise = (RGB_train.T + 0.1*np.mean(RGB_train)*np.random.normal(size=(self.Ntrain,3))).reshape(-1)
            np.random.seed()
            
            t1 = time.time()
            if self.verbose:
                print('\n### Iodine Thickness Reconstruction', end='', flush=True)
            d_reconstr = self.iodine_thickness_reconstruction(RGB_noise,
                                                              wvl_hist_temp,
                                                              wvl_bin_centers)
            t2 = time.time()
            if self.verbose:
                print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)

            fom = np.sum((d_reconstr[:,-1] - self.training_data[:,-1])**2)
        
        elif self.FoM_type == 'K_edge_imaging_v3':
            RGB_train, _, _ = self.ground_truth_forward_model(self.train_thickness.reshape(self.Nsample,-1).T, wvl_hist_temp, wvl_bin_centers)
            RGB_test, _, _ = self.ground_truth_forward_model(self.test_thickness.reshape(self.Nsample,-1).T, wvl_hist_temp, wvl_bin_centers)
            
            np.random.seed(0)
            RGB_train_noise = np.tile(RGB_train.T, (10,1))
            RGB_test_noise = np.tile(RGB_test.T, (10,1))
            for i in range(3):
                RGB_train_noise[:,i] += 0.1*np.mean(RGB_train[i,:])*np.random.normal(size=10*self.Nx_img_train*self.Nz_img_train)
                RGB_test_noise[:,i] += 0.1*np.mean(RGB_test[i,:])*np.random.normal(size=10*self.Nx_img_test*self.Nz_img_test)
            np.random.seed()
            
            train_thickness_tile = np.tile(np.log10(self.train_thickness.reshape(self.Nsample,-1).T), (10,1))
            test_thickness_tile = np.tile(np.log10(self.test_thickness.reshape(self.Nsample,-1).T), (10,1))
            
            if gamma_optimization:
                if rank == 0:
                    t1 = time.time()
                    if self.verbose:
                        print('\n### Network Training', flush=True)
                    self.model = dnn.train(train_thickness_tile,
                                           test_thickness_tile,
                                           RGB_train_noise,
                                           RGB_test_noise,
                                           self.Nsample,
                                           self.energy_bins,
                                           3)
                    torch.save(self.model.state_dict(), directory + "/data/dnn_state_" + self.identifier + ".pt")
                    t2 = time.time()
                    if self.verbose:
                        print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                else:
                    while True:
                        if os.path.exists(directory + "/data/dnn_state_" + self.identifier + ".pt"):
                            break
                        time.sleep(5)
                        
                    self.model = model = torch.nn.Sequential(torch.nn.Linear(self.Nsample, self.energy_bins),
                                                             torch.nn.ReLU(),
                                                             torch.nn.Linear(self.energy_bins, 3)).to(device)
                    self.model.load_state_dict(torch.load(directory + "/data/dnn_state_" + self.identifier + ".pt"))
                
                t1 = time.time()
                if self.verbose:
                    print('\n### Element Thickness Reconstruction', end='', flush=True)
                d_train, loss_train, d_test, loss_test = dnn.predict(self.model,
                                                                     sensRGB_interp,
                                                                     wvl_hist_temp,
                                                                     self.gamma_block1_opt,
                                                                     self.energy_bins,
                                                                     self.Ntrain,
                                                                     self.raw_clip)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                fom = loss_test
            
            else:
                t1 = time.time()
                if self.verbose:
                    print('\n### Element Thickness Reconstruction', end='', flush=True)
                d_train, loss_train, d_test, loss_test = dnn.predict(self.model,
                                                                     sensRGB_interp,
                                                                     wvl_hist_temp,
                                                                     self.gamma_block1_opt,
                                                                     self.energy_bins,
                                                                     self.Ntrain,
                                                                     self.raw_clip)
                t2 = time.time()
                if self.verbose:
                    print('(' + str(np.round(t2-t1,2)) + 's)', flush=True)
                
                fom = loss_test
        
        if rank == 0:
            if self.FoM_type == 'N_energy_bins':
                np.savez(directory + "/results/RGB_data_" + self.identifier, wvl_hist_all=wvl_hist_all, wvl_bin_centers=wvl_bin_centers,
                         RGB=RGB, fom=fom, sensRGB=sensRGB_interp)
            elif self.FoM_type == 'K_edge_imaging':
                if gamma_optimization:
                    np.savez(directory + "/results/reg_matrix_" + self.identifier, gamma_block1=self.gamma_block1_opt, gamma_block2=self.gamma_block2_opt,
                             I_reconstr=I_reconstr, I_image_reconstr=I_image_reconstr, d_reconstr=d_reconstr, d_image_reconstr=d_image_reconstr, fom=fom,
                             xraySrc_fwd=xraySrc_fwd, linAttCoeff=self.linAttCoeff, training_data=self.training_data, sampled_data=self.sampled_data,
                             scale_thickness=self.scale_thickness, xraySrc=self.xraySrc, RGB_image=RGB_image.reshape(3, self.Nx_img, self.Nz_img))
                else:
                    np.savez(directory + "/results/pso_data_" + self.identifier, gamma_block1=self.gamma_block1_opt, wvl_hist_temp=wvl_hist_temp, wvl_hist_all=wvl_hist_all,
                             wvl_bin_centers=wvl_bin_centers, RGB_train=RGB_train.T.reshape(-1), fom=fom, I_reconstr=I_reconstr, I_image_reconstr=I_image_reconstr,
                             training_data=self.training_data, xraySrc_fwd=xraySrc_fwd, RGB_noise=RGB_noise, sampled_data=self.sampled_data, RGB_image=RGB_image.reshape(3, self.Nx_img, self.Nz_img),
                             linAttCoeff=self.linAttCoeff, scale_thickness=self.scale_thickness, xraySrc=self.xraySrc)
            elif self.FoM_type == 'K_edge_imaging_v2':
                np.savez(directory + "/results/svd_method_" + self.identifier, d_reconstr=d_reconstr, fom=fom, xraySrc_fwd=xraySrc_fwd, linAttCoeff=self.linAttCoeff,
                         linAttCoeff_svd=self.linAttCoeff_svd, training_data=self.training_data, element_thickness_per_pixel=self.element_thickness_per_pixel,
                         xraySrc=self.xraySrc)
            elif self.FoM_type == 'K_edge_imaging_v3':
                np.savez(directory + "/results/pso_data_" + self.identifier, wvl_hist_temp=wvl_hist_temp, wvl_hist_all=wvl_hist_all, wvl_bin_centers=wvl_bin_centers,
                         RGB_train=RGB_train.T, RGB_test=RGB_test.T, loss_train=loss_train, loss_test=loss_test, d_train=d_train, d_test=d_test, train_thickness=train_thickness_tile,
                         test_thickness=test_thickness_tile, xraySrc_fwd=xraySrc_fwd, RGB_train_noise=RGB_train_noise, RGB_test_noise=RGB_test_noise, linAttCoeff=self.linAttCoeff,
                         xraySrc=self.xraySrc)
        
        return fom
    
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
            
        RGB_unclip = sensRGB_interp.T @ wvl_hist_all
        RGB = RGB_unclip.copy()
        # RGB /= self.raw_clip
        RGB /= np.max(RGB)
        
        return RGB, sensRGB_interp
    
    def ground_truth_forward_model(self, d_sample, wvl_hist_temp, wvl_bin_centers):
        d_sample = d_sample.reshape(-1,self.Nsample)
        linAttCoeff_fwd = self.linAttCoeff @ d_sample.T # energy_bins x Ntrain
        xraySrc_fwd = self.xraySrc[:,np.newaxis]*np.exp(-linAttCoeff_fwd)
        wvl_hist_all = wvl_hist_temp @ xraySrc_fwd
        
        RGB_fwd = self.RGB_compute(wvl_hist_all, wvl_bin_centers)
        
        return RGB_fwd, wvl_hist_all, xraySrc_fwd
    
#    def transmitted_intensity_reconstruction(self, RGB_train, wvl_hist_temp, wvl_bin_centers, gamma_block=None, nfev=50):
#        np.random.seed(0)
#        RGB_noise = (RGB_train.T + 0.1*np.mean(RGB_train)*np.random.normal(size=(self.Ntrain,3))).reshape(-1)
#        
#        sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
#        for i in range(3):
#            sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
#            sensRGB_interp[:,i] /= 100
#        Di = sensRGB_interp.T @ wvl_hist_temp
#        D = np.zeros((self.Ntrain*3, self.Ntrain*self.energy_bins))
#        for i in range(self.Ntrain):
#            D[3*i:3*(i+1),self.energy_bins*i:self.energy_bins*(i+1)] = Di
#        D /= self.raw_clip
#        
#        gamma = np.zeros((self.Ntrain*self.energy_bins, self.Ntrain*self.energy_bins))
#        if gamma_block is None:
#            weights = np.zeros(self.energy_bins)
#            
#            E_norm1 = self.energy_list[self.energy_list <= (self.energy_list[0] + self.energy_list[-1])/2]
#            E_norm1 = (E_norm1 - E_norm1[0])/((self.energy_list[0] + self.energy_list[-1])/2 - E_norm1[0])
#            weights[:E_norm1.size] = np.interp(E_norm1, np.linspace(0, 1, 11), np.linspace(1, 0, 11))
#            
#            E_norm2 = self.energy_list[self.energy_list >= (self.energy_list[0] + self.energy_list[-1])/2]
#            E_norm2 = (E_norm2 - (self.energy_list[0] + self.energy_list[-1])/2)/(E_norm2[-1] - (self.energy_list[0] + self.energy_list[-1])/2)
#            try:
#                weights[E_norm1.size-1:] = np.interp(E_norm2, np.linspace(0, 1, 11), np.linspace(0, 1, 11))
#            except:
#                weights[E_norm1.size:] = np.interp(E_norm2, np.linspace(0, 1, 11), np.linspace(0, 1, 11))
#        
#            gamma_block = np.diag(weights)
#        
#        for i in range(self.Ntrain):
#            gamma[self.energy_bins*i:self.energy_bins*(i+1),self.energy_bins*i:self.energy_bins*(i+1)] = gamma_block
#        
#        # gamma = np.identity(self.Ntrain*self.energy_bins)
#        
#        if self.Ntrain*self.energy_bins < 1e5:
#            # Direct Inversion
#            t1 = time.time()
#            I_reconstr = np.linalg.inv(D.T @ D + gamma) @ D.T @ RGB_noise
#            t2 = time.time()
#            if self.verbose:
#                print('\n### Transmitted X-Ray Spectrum Reconstruction (' + str(np.round(t2-t1,2)) + 's)', flush=True)
#        else:
#            # Iterative Approach
#            ub = np.tile(self.xraySrc, self.Ntrain)
#            bnd = Bounds(lb=0*np.ones(self.Ntrain*self.energy_bins), ub=ub, keep_feasible=False)
#            if self.verbose:
#                print('\n### Transmitted X-Ray Spectrum Reconstruction', flush=True)
#            reconstr_result = least_squares(self.intensity_lsqr_obj, x0=np.min((np.min(ub),1e-6))*np.ones(self.Ntrain*self.energy_bins),
#                                            jac=self.jac_intensity_lsqr_obj, bounds=bnd, max_nfev=nfev,
#                                            verbose=2 if self.verbose else 0, args=(D, RGB_noise, gamma))
#            I_reconstr = reconstr_result.x
#        
#        return I_reconstr, D, RGB_noise, gamma
        
    def intensity_lsqr_obj(self, I_reconstr, D, RGB_noise, gamma):
        res = D.T @ RGB_noise - (D.T @ D + gamma) @ I_reconstr
        
        return res
    
    def jac_intensity_lsqr_obj(self, I_reconstr, D, RGB_noise, gamma):
        jac = -(D.T @ D + gamma)
        
        return jac
    
    def regularization_matrix_optimization1(self, RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, gamma_block=None):
#        if gamma_block is None:
#            gamma_block = np.identity(self.energy_bins)
#        gamma_vec = gamma_block.reshape(-1)
        
        xraySrc_fwd = xraySrc_fwd.T.reshape(-1)
        
        self.loss_hist = np.zeros(0)
        self.n_iter = 0
        
#        beta1 = 0.9
#        beta2 = 0.999
#        eta = 0.01
#        maxiter = 100
#        
#        # ADAM Optimization
#        n_iter = 0
#        jac_mean = np.zeros(gamma_diag.size)
#        jac_var = np.zeros(gamma_diag.size)
#        gamma_diag_fin = gamma_diag.copy()
#        loss_prev = 100
#        loss_hist = np.zeros(0)
#        while True:
#            n_iter += 1
#
#            # Get Gradient
#            loss, D, RGB_noise, gamma = self.step1_reg_obj(gamma_diag, RGB_train, wvl_hist_temp, wvl_bin_centers, xraySrc_fwd)
#            jac = self.jac_step1_reg_obj(gamma, D, RGB_noise)
#            
#            # Finite Difference Check
#            jac_FD = np.zeros(self.energy_bins**2)
#            for i in range(self.energy_bins**2):
#                print('Finite Difference ' + str(i), flush=True)
#                gamma_diag_temp = gamma_diag.copy()
#                gamma_diag_temp[i] -= 1e-6
#                loss1, _, _, _ = self.step1_reg_obj(gamma_diag_temp, RGB_train, wvl_hist_temp, wvl_bin_centers, xraySrc_fwd)
#                gamma_diag_temp = gamma_diag.copy()
#                gamma_diag_temp[i] += 1e-6
#                loss2, _, _, _ = self.step1_reg_obj(gamma_diag_temp, RGB_train, wvl_hist_temp, wvl_bin_centers, xraySrc_fwd)
#                jac_FD[i] = (loss2 - loss1)/2e-6
#            
#            np.savez(directory + '/data/debug_reg_gradient', gamma_diag=gamma_diag, jac=jac, jac_FD=jac_FD)
#            assert False
#            
#            loss_hist = np.append(loss_hist, loss)
#            if loss < loss_prev:
#                gamma_diag_fin = gamma_diag.copy()
#                loss_prev = loss
#            
#            if self.verbose:
#                print('%13s%19s' %('Iteration', 'Loss'), flush=True)
#                print('%13d%19.6f' %(n_iter, loss))
#            
#            # Update Average Gradients
#            jac_mean = beta1*jac_mean + (1 - beta1)*jac
#            jac_var = beta2*jac_var + (1 - beta2)*jac**2
#            
#            # Unbias Average Gradients
#            jac_mean_unbiased = jac_mean/(1 - beta1**n_iter)
#            jac_var_unbiased = jac_var/(1 - beta2**n_iter)
#            
#            # Update Variables
#            gamma_diag -= eta*jac_mean_unbiased/(np.sqrt(jac_var_unbiased) + 1e-8)
#            # adam_eta *= 0.95
#            
#            if n_iter > maxiter:
#                break
#        
#        np.savez(directory + '/data/adam_trajectory_' + self.identifier, loss_hist=loss_hist)
        
        if self.verbose:
            print('%13s%19s' %('Iteration', 'Loss'), flush=True)
        
#        if gamma_block is not None:
#            gamma_vec = gamma_block.reshape(-1)
#            result = minimize(self.step1_reg_obj, x0=gamma_vec, args=(RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, self.energy_bins, self.Ntrain, self.raw_clip),
#                              method='BFGS', jac=jit_fct.jac_step1_reg_obj, options={'maxiter':self.bfgs_maxiter})
#        else:

        if gamma_block is None:
            gamma_diag = np.ones(self.energy_bins)
        else:
            gamma_diag = np.diag(gamma_block)
        
#        jac = jit_fct.jac_step1_reg_obj_diag(gamma_diag, RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, self.energy_bins, self.Ntrain, self.raw_clip)
#        jac_fd = np.zeros(self.energy_bins)
#        for i in range(self.energy_bins):
#            gamma_diag_temp = gamma_diag.copy()
#            gamma_diag_temp[i] -= 1e-6
#            loss1 = self.step1_reg_obj_diag(gamma_diag_temp, RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, self.energy_bins, self.Ntrain, self.raw_clip)
#            
#            gamma_diag_temp = gamma_diag.copy()
#            gamma_diag_temp[i] += 1e-6
#            loss2 = self.step1_reg_obj_diag(gamma_diag_temp, RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, self.energy_bins, self.Ntrain, self.raw_clip)
#            
#            jac_fd[i] = (loss2 - loss1)/2e-6
#        if comm.rank == 0:
#            np.savez(directory + "/data/finite_diff_check", jac=jac, jac_fd=jac_fd)
#        assert False
        
        result = minimize(self.step1_reg_obj_diag, x0=gamma_diag, args=(RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, self.energy_bins, self.Ntrain, self.raw_clip),
                          method='BFGS', jac=jit_fct.jac_step1_reg_obj_diag, options={'maxiter':self.bfgs_maxiter})
        
        np.savez(directory + '/data/BFGS_trajectory_reg1_' + self.identifier, loss_hist=self.loss_hist)
        
        # return result.x.reshape(self.energy_bins, self.energy_bins)
        return np.diag(result.x)
    
    def step1_reg_obj(self, gamma_vec, RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, energy_bins, Ntrain, raw_clip):
        I_reconstr, D, gamma = jit_fct.transmitted_intensity_reconstruction(RGB_noise,
                                                                            sensRGB_interp,
                                                                            wvl_hist_temp,
                                                                            gamma_vec.reshape(energy_bins, energy_bins),
                                                                            energy_bins,
                                                                            Ntrain,
                                                                            raw_clip)

        xraySrc_temp = xraySrc_fwd.reshape(Ntrain, energy_bins)
        weights = np.tile(1/np.sum(xraySrc_temp**2, axis=1), (energy_bins,1)).T.reshape(-1)
        loss = np.sum((I_reconstr - xraySrc_fwd)**2)
        
        self.n_iter += 1
        if self.verbose:
            print('%13d%19.6f' %(self.n_iter, loss), flush=True)
            
        self.loss_hist = np.append(self.loss_hist, loss)
        
        return loss
    
    def step1_reg_obj_diag(self, gamma_diag, RGB_noise, sensRGB_interp, wvl_hist_temp, xraySrc_fwd, energy_bins, Ntrain, raw_clip):
        I_reconstr, D, gamma = jit_fct.transmitted_intensity_reconstruction(RGB_noise,
                                                                            sensRGB_interp,
                                                                            wvl_hist_temp,
                                                                            np.diag(gamma_diag),
                                                                            energy_bins,
                                                                            Ntrain,
                                                                            raw_clip)
        
        xraySrc_temp = xraySrc_fwd.reshape(Ntrain, energy_bins)
        weights = np.tile(1/np.sum(xraySrc_temp**2, axis=1), (energy_bins,1)).T.reshape(-1)
        loss = np.sum(weights*(I_reconstr - xraySrc_fwd)**2)

        self.n_iter += 1
        if self.verbose:
            print('%13d%19.6f' %(self.n_iter, loss), flush=True)
            
        self.loss_hist = np.append(self.loss_hist, loss)
        
        return loss
    
#    def jac_step1_reg_obj(self, gamma_vec, RGB_train, wvl_hist_temp, wvl_bin_centers, xraySrc_fwd):
#        I_reconstr, D, RGB_noise, gamma = self.transmitted_intensity_reconstruction(RGB_train, wvl_hist_temp, wvl_bin_centers, gamma_block=gamma_vec.reshape(self.energy_bins, self.energy_bins), nfev=50)
#    
#        A1 = D.T @ D + gamma
#        if self.Ntrain*self.energy_bins < 1e5:
#            A1_inv = np.linalg.inv(A1)
#        grad = np.zeros((self.Ntrain*self.energy_bins, self.energy_bins, self.energy_bins))
#        
#        for i in range(self.energy_bins):
#            for j in range(self.energy_bins):
#                P = np.zeros((self.energy_bins, self.energy_bins))
#                P[i,j] = 1
#                P_block_diag = np.zeros((self.Ntrain*self.energy_bins, self.Ntrain*self.energy_bins))
#                for k in range(self.Ntrain):
#                    P_block_diag[self.energy_bins*k:self.energy_bins*(k+1),self.energy_bins*k:self.energy_bins*(k+1)] = P
#
#                if self.Ntrain*self.energy_bins < 1e5:
#                    result_vector = -A1_inv @ P_block_diag @ A1_inv @ D.T @ RGB_noise
#                else:
#                    A2 = A1 @ P_block_diag.T @ A1
#        
#                    def fun(x, D, A, RGB_noise):
#                        return A @ x + D.T @ RGB_noise
#                    
#                    def jac(x, D, A, RGB_noise):
#                        return A
#                
#                    invert_result = least_squares(fun, x0=np.zeros(self.Ntrain*self.energy_bins),
#                                                  jac=jac, max_nfev=50,
#                                                  verbose=2 if self.verbose else 0, args=(D, A2, RGB_noise))
#                    result_vector = invert_result.x
#            
#                grad[:,i,j] += result_vector
#        
#        d_loss = 2*(I_reconstr - xraySrc_fwd)
#        grad_loss = d_loss.reshape(1,-1) @ grad.reshape(self.Ntrain*self.energy_bins, self.energy_bins**2)
#        
#        return grad_loss.reshape(-1)
    
    def regularization_matrix_optimization2(self, T_reconstr, gamma_block=None):
        if gamma_block is None:
            gamma_diag = np.ones(self.Nsample)
        else:
            gamma_diag = np.diag(gamma_block)
        # gamma_vec = gamma_block.reshape(-1)
        
        training_data = self.training_data.reshape(-1) # [:,-1] # only compute for iodine
        
        self.loss_hist = np.zeros(0)
        self.n_iter = 0

        if self.verbose:
            print('%13s%19s' %('Iteration', 'Loss'), flush=True)

#        jac = jit_fct.jac_step2_reg_obj_diag(gamma_diag, T_reconstr, self.linAttCoeff, training_data, self.energy_bins, self.Nsample, self.Ntrain)
#        jac_fd = np.zeros(self.Nsample)
#        for i in range(self.Nsample):
#            gamma_diag_temp = gamma_diag.copy()
#            gamma_diag_temp[i] -= 1e-6
#            loss1 = self.step2_reg_obj_diag(gamma_diag_temp, T_reconstr, self.linAttCoeff, training_data, self.energy_bins, self.Nsample, self.Ntrain)
#            
#            gamma_diag_temp = gamma_diag.copy()
#            gamma_diag_temp[i] += 1e-6
#            loss2 = self.step2_reg_obj_diag(gamma_diag_temp, T_reconstr, self.linAttCoeff, training_data, self.energy_bins, self.Nsample, self.Ntrain)
#            
#            jac_fd[i] = (loss2 - loss1)/2e-6
#        if comm.rank == 0:
#            np.savez(directory + "/data/finite_diff_check", jac=jac, jac_fd=jac_fd)
#        assert False

        result = minimize(self.step2_reg_obj_diag, x0=gamma_diag, args=(T_reconstr, self.linAttCoeff, training_data, self.energy_bins, self.Nsample, self.Ntrain),
                          method='BFGS', jac=jit_fct.jac_step2_reg_obj_diag, options={'maxiter':self.bfgs_maxiter})
        
        np.savez(directory + '/data/BFGS_trajectory_reg2_' + self.identifier, loss_hist=self.loss_hist)
        
        # return result.x.reshape(self.Nsample, self.Nsample)
        return np.diag(result.x)
    
    def step2_reg_obj(self, gamma_vec, T_reconstr, linAttCoeff, training_data, energy_bins, Nsample, Ntrain):
        d_reconstr, mu, gamma = jit_fct.element_thickness_reconstruction(T_reconstr,
                                                                         linAttCoeff,
                                                                         gamma_vec.reshape(Nsample, Nsample),
                                                                         energy_bins,
                                                                         Nsample,
                                                                         Ntrain)
                                                                             
        # loss = np.sum(((d_reconstr.reshape(self.Ntrain, self.Nsample)[:,-1] - training_data)**2)/training_data) # only compute for iodine
        weights = 1/training_data
        weights[training_data==0] = np.max(weights[weights!=np.inf])
        weights[(np.arange(weights.size)%Nsample)!=Nsample-1] = 0
        loss = np.sum((d_reconstr - training_data)**2)
        
        self.n_iter += 1
        if self.verbose:
            print('%13d%19.6f' %(self.n_iter, loss), flush=True)
            
        self.loss_hist = np.append(self.loss_hist, loss)
        
        return loss
    
    def step2_reg_obj_diag(self, gamma_diag, T_reconstr, linAttCoeff, training_data, energy_bins, Nsample, Ntrain):
        d_reconstr, mu, gamma = jit_fct.element_thickness_reconstruction(T_reconstr,
                                                                         linAttCoeff,
                                                                         np.diag(gamma_diag),
                                                                         energy_bins,
                                                                         Nsample,
                                                                         Ntrain)
                                                                             
        loss = np.sum((d_reconstr - training_data)**2)
        
        self.n_iter += 1
        if self.verbose:
            print('%13d%19.6f' %(self.n_iter, loss), flush=True)
            
        self.loss_hist = np.append(self.loss_hist, loss)
        
        return loss
    
    def iodine_thickness_reconstruction(self, RGB_noise, wvl_hist_temp, wvl_bin_centers):
        RGB_noise = RGB_noise.reshape(self.Ntrain, 3)

        d_reconstr = np.zeros((self.Ntrain, 3))
        bnd = Bounds(lb=np.array([-np.inf,-np.inf,0]), ub=np.inf, keep_feasible=False)
        for nt in range(self.Ntrain):
            if self.verbose:
                print('\t | Training Sample ' + str(nt), end='', flush=True)
                
            fun_best = 1e3
            x_best = np.zeros(3)
            
            for i in range(10):
                x0 = np.random.normal(scale=0.1, size=3)
                x0[-1] = 0
                    
                self.loss_hist = np.zeros(0)
                self.n_iter = 0
                result = minimize(self.iodine_thickness_obj, x0=x0, args=(RGB_noise[nt,:], wvl_hist_temp, wvl_bin_centers),
                                  method='L-BFGS-B', jac=self.jac_iodine_thickness_obj, bounds=bnd, options={'maxiter':self.bfgs_maxiter})
                
                if result.fun < fun_best:
                    x_best = result.x
                
                if self.verbose:
                    print(' | ' + str(result.fun), end='', flush=True)
                    
            d_reconstr[nt,:] = x_best
            
            if self.verbose:
                print('')
        
        return d_reconstr

    def iodine_thickness_obj(self, d_reconstr, RGB_noise, wvl_hist_temp, wvl_bin_centers):
        linAttCoeff_fwd = self.linAttCoeff_svd @ d_reconstr
        xraySrc_fwd = self.xraySrc*np.exp(-linAttCoeff_fwd)
        wvl_hist_all = wvl_hist_temp @ xraySrc_fwd
        
        RGB_reconstr = self.RGB_compute(wvl_hist_all, wvl_bin_centers)
        
        loss = np.sum((RGB_reconstr.reshape(-1) - RGB_noise)**2)
        
        self.n_iter += 1
            
        self.loss_hist = np.append(self.loss_hist, loss)
        
        return loss
        
    def jac_iodine_thickness_obj(self, d_reconstr, RGB_noise, wvl_hist_temp, wvl_bin_centers):
        linAttCoeff_fwd = self.linAttCoeff_svd @ d_reconstr
        xraySrc_fwd = self.xraySrc*np.exp(-linAttCoeff_fwd)
        wvl_hist_all = wvl_hist_temp @ xraySrc_fwd
        
        RGB_reconstr = self.RGB_compute(wvl_hist_all, wvl_bin_centers)
    
        sensRGB_interp = np.zeros((wvl_bin_centers.size, 3))
        for i in range(3):
            sensRGB_interp[:,i] = np.interp(wvl_bin_centers, self.sensRGB[:,0], self.sensRGB[:,i+1])
            sensRGB_interp[:,i] /= 100
        
        jac = -(1/self.raw_clip)*sensRGB_interp.T @ wvl_hist_temp @ np.diag(self.xraySrc) @ np.diag(np.exp(-self.linAttCoeff_svd @ d_reconstr)) @ self.linAttCoeff_svd
        grad = 2*(RGB_reconstr.reshape(-1) - RGB_noise).reshape(1,-1) @ jac
        
        return grad.reshape(-1)
    
    def make_simulation_macro(self, rank, thickness, source_energy):
        if os.path.exists(self.G4directory + "/build/output" + str(rank) + ".root"):
             os.remove(self.G4directory + "/build/output" + str(rank) + ".root")
        
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
            
            # mac.write("/run/numberOfThreads " + str(int(self.Nthreads)) + "\n")
            mac.write("/run/initialize\n")
            # mac.write("/control/execute vis.mac\n\n")
            
            mac.write("/gun/momentumAmp " + str(np.round(source_energy, 6)) + " keV\n")
            mac.write("/gun/position 0. 0. " + str(np.round(-dzWorld/2.02, 6)) + " cm\n")
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
            if self.verbose:
                print('\t | Energy: ' + str(self.energy_list[n]) + 'keV (' + str(t2 - t1) + ' s)', flush=True)
        
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
    np.random.seed(0)
    Ntrain = 100
    training_data = np.zeros((Ntrain, 3))
    training_data[:,:2] = 2*(np.random.rand(Ntrain, 2) - 1)
    training_data[:,2] = 3*(np.random.rand(Ntrain) - 1) - 3
    training_data = 10**training_data
    training_data *= 2/np.sum(training_data, axis=1)[:,np.newaxis]

    simulation = geant4(G4directory = "/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main",
                        dimScint = np.array([1,1]), # in cm
                        dimDet = np.array([1,1]), # in cm
                        gap = 10, # in cm
                        structureID = 1,
                        Nlayers = 3,
                        scintillator_material = np.array([2,4,3]), # 1:YAGCe, 2:ZnSeTe, 3:LYSOCe, 4:CsITl
                        Ngrid = np.array([1,1]),
                        detector_thickness = 0.1, # in um
                        check_overlap = 1,
                        energy_range = np.array([10,80]), # in keV
                        energy_bins = 15,
                        Nphoton = 10000,
                        wvl_bins = 801,
                        source_type = 'xray_tube', # uniform
                        xray_target_material = 'W',
                        xray_operating_V = 80, # in keV
                        xray_tilt_theta = 12, # in deg
                        xray_filter_material = 'Al',
                        xray_filter_thickness = 0.3, # in cm
                        imaging_samples = np.array(['Bone','Blood','I']),
                        training_data = training_data,
                        FoM_type = 'K_edge_imaging', # K_edge_imaging / N_energy_bins
                        verbose = True
                        )
    
    simulation.cmake()
    simulation.make()
    
    fom = simulation.FoM(0, thickness=np.array([0.527273,0.809091,0.184700])) # in mm
    print("FoM: " + str(fom))
    
    # wvl_hist, wvl_bin = simulation.analyze_all_energies(0, thickness=np.array([0.527273,0.809091,0.184700]), pht_per_energy=simulation.Nphoton*np.ones(simulation.energy_bins))