# Reference: Tucker et al. "Semiempirical model for generating tungsten target x-ray spectra" (1991)

import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
import util.read_txt as txt

def get_spectrum(target_material,
                 E, # xray spectrum energy bins (keV)
                 T0, # incident electron energy (keV)
                 theta, # xray target tilt angle (deg)
                 filter_material,
                 d_filter, # filter thickness (cm)
                 ):

    e0 = 1.602e-19 # elementary charge (C)
    m0 = 9.11e-31 # electron rest mass (kg)
    c = 299792458 # (m/s)
    eps0 = 8.85e-12 # vacuum permittivity (F/m)
    h = 6.63e-34 # Planck constant (J s)
    alpha = e0**2/(2*eps0*h*c) # fine structure constant (unitless)
    re = e0**2/(4*np.pi*eps0*m0*c**2) # classical electron radius (m)
    
    m0 *= 1e3
    c *= 1e2
    re *= 1e2
    
    if target_material == 'W':
        rho = 19.28 # target density (g/cm3)
        c_TW_raw = np.array([[50,5.4e5],
                             [75,6.25e5],
                             [100,7e5],
                             [150,8.4e5],
                             [200,1e6]])
        c_TW = np.interp(T0, c_TW_raw[:,0], c_TW_raw[:,1]) # Thomas-Whiddington constant (keV2 cm2 / g)
        
        A0 = 3.685e-2 # units of photons/electron
        A1 = 2.900e-5 # units of photons/keV
        B1 = -5.049
        B2 = 10.847
        B3 = -10.516
        B4 = 3.842
        
        A_msp = 2024.1 # (keV cm2 / g)
        B_msp = 10361 # (keV cm2 / g)
        C_msp = 0.04695 # (/keV)
        
        Z = 74 # atomic number
        A = 183.84*1.66e-24 # atomic mass (g/atom)
        
        EK = 69.5 # K-shell binding energy (keV) = absorption edge location
        Ec = np.array([59.32,57.98,67.2]) # characteristic emission energies (keV) --> 67.1 has been omitted (b/c it's too close to 67.2)
        fEc = np.array([0.45,0.2592,0.1521]) # fractional yield of each characteristic emission energy (0.0387 corresponds to 67.1 keV)
        
        AK = 1.349e-3 # (photons/electron)
        nK = 1.648
    
    T = np.linspace(1e-3, T0, int(1e5))
    if T0 > EK:
        for i in range(Ec.size):
            E = np.delete(E, np.argmin(np.abs(E - Ec[i])))
        E = np.sort(np.hstack((E, Ec)))
    E_mesh, T_mesh = np.meshgrid(E, T, indexing='ij')
    
    mu_raw = txt.read_txt(directory + '/data/linAttCoeff' + target_material)
    mu = 10**np.interp(np.log10(E), np.log10(mu_raw[:,0]), np.log10(mu_raw[:,1])) # target linear attenuation coefficient (/cm)
    mu_Ec = 10**np.interp(np.log10(Ec), np.log10(mu_raw[:,0]), np.log10(mu_raw[:,1]))
    
    mu_filter = np.zeros((E.size, len(filter_material)))
    for i in range(len(filter_material)):
        mu_raw = txt.read_txt(directory + '/data/linAttCoeff' + filter_material[i])
        mu_filter[:,i] = 10**np.interp(np.log10(E), np.log10(mu_raw[:,0]), np.log10(mu_raw[:,1]))
    
    theta *= np.pi/180

    ### Bremsstrahlung Radiation
    
    # (1) Fraction of photons that exits along target path (accounting for target attenuation)
    F = np.exp(-mu[:,np.newaxis]*(T0**2 - T_mesh**2)/(rho*c_TW*np.sin(theta)))
    
    # (2) B (function of electron energy)
    B = (A0 + A1*T0)*(1 + B1*(E_mesh/T_mesh) + B2*(E_mesh/T_mesh)**2 + B3*(E_mesh/T_mesh)**3 + B4*(E_mesh/T_mesh)**4)
    B[E_mesh>T_mesh] = 0
    
    # (3) Mass stopping term
    msp = A_msp + B_msp*np.exp(-T_mesh*C_msp)
    
    # (4) Number of X-Ray Photons Emitted per Electron per keV (photons/electron keV)
    integrand = ((B*(T_mesh + m0*c**2))/T_mesh)*F/msp
    integral = np.sum(0.5*(integrand[:,:-1] + integrand[:,1:])*(T[1] - T[0]), axis=1)
    N_B = ((alpha*re**2*Z**2)/(A*E))*integral
    
    ### Characteristic Radiation
    if T0 > EK:
        # (1) X-Ray production probability
        R = (T0**2 - EK**2)/(rho*c_TW)
        x = np.linspace(0, R, int(1e5))
        
        mu_Ec_mesh, x_mesh = np.meshgrid(mu_Ec, x, indexing='ij')
        
        P = (3/2)*(1 - (x_mesh/R)**2)
        
        # (2) Number of X-Ray Photons Emitted per Electron per keV (photons/electron keV)
        integrand = P*np.exp(-mu_Ec_mesh*x_mesh/np.sin(theta))
        integral = np.sum(0.5*(integrand[:,:-1] + integrand[:,1:])*(x[1] - x[0]), axis=1)
        N_C = AK*(T0/EK - 1)**nK*fEc*integral
    
        ### Overall Spectrum
        for i in range(Ec.size):
            N_B[E==Ec[i]] += N_C[i]
    
    ### Inherent Filtering
    N = N_B.copy()
    for i in range(len(filter_material)):
        N *= np.exp(-mu_filter[:,i]*d_filter[i])
    
    return E, N

# factor in filtration
if __name__ == '__main__':
#    energy_distribution = np.load(directory + "/data/energy_distribution_data_ZnSe_Gadox_NaI_67keV_100um.npz")['energy_distribution']
#    A = np.sum(energy_distribution, axis=(0,1))
#
#    I_bin = np.zeros((19, 19, 3))
#    for i in range(19):
#        for j in range(19):
#            print(str(i) + ' | ' + str(j))
#            E, N = get_spectrum('W', # target material
#                                np.linspace(16, 67, 52), # xray energy bins (keV)
#                                67, # incident electron energy (keV)
#                                12, # target tilt angle (deg)
#                                ['Sn','Al'], # filter material
#                                [0.01+i*0.005,0.1+j*0.05], # filter thickness (cm)
#                                )
#            
#            A_interp = np.interp(E, np.linspace(16, 67, 52), A)
#            N *= A_interp
#            I_bin[i,j,0] = np.trapz(N[E<=33], E[E<=33])
#            I_bin[i,j,1] = np.trapz(N[(E>33)*(E<=50)], E[(E>33)*(E<=50)])
#            I_bin[i,j,2] = np.trapz(N[E>50], E[E>50])
#        
#    I_bin /= np.sum(I_bin, axis=2)[:,:,np.newaxis]
#    mse = np.sqrt(np.sum((I_bin - 1/3)**2, axis=2))
#                        
#    np.savez(directory + '/data/simulated_xray_spectrum_sweep', E=E, N=N, I_bin=I_bin, mse=mse)
    
    E, N = get_spectrum('W', # target material
                        np.linspace(16, 67, 52), # xray energy bins (keV)
                        67, # incident electron energy (keV)
                        12, # target tilt angle (deg)
                        ['Al'], # filter material
                        [0.3], # filter thickness (cm)
                        )

    np.savez(directory + '/data/simulated_xray_spectrum', E=E, N=N)