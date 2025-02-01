import sys
sys.path.insert(0, '../')
import numpy as np
from utils.utils import *
import random
import scipy
from design.hybrid_scintillator import *
import torch

simulation_template = \
"""
/structure/xWorld {worldSizeXY}
/structure/yWorld {worldSizeXY}
/structure/zWorld {worldSizeZ}

/structure/isMultilayerNS 1
/structure/nLayersNS {nLayersNS}
/structure/substrateThickness {substrateThickness}
/structure/scintillatorThickness {scintillatorThickness}
/structure/dielectricThickness {dielectricThickness}

/structure/constructDetectors 0
/structure/nGridX 1
/structure/nGridY 1
/structure/detectorDepth 0.1e-3

/structure/checkDetectorsOverlaps 0

/run/initialize

/run/beamOn {numEvents}
"""

def scint_thickness_to_end(z, nLayersNS, substrateThickness, dielectricThickness, scintillatorThickness):
    z -= substrateThickness
    distance_to_begin = 0
    total_thickness = dielectricThickness * (nLayersNS - 1)/2
    for l in range(int((nLayersNS - 1) / 2)):
        if (z > dielectricThickness):
            z -= dielectricThickness
            distance_to_begin += dielectricThickness
        else:
            if (z > 0):
                distance_to_begin += z
            break
        z -= scintillatorThickness
    distance_to_end = total_thickness - distance_to_begin
    return distance_to_end

def k_to_absorption_coeff(k):
    lambda_0 = 430e-6
    abs_coeff = 4 * np.pi * k / lambda_0
    return abs_coeff

def abs_to_the_end(z, k, nLayersNS, substrateThickness, dielectricThickness, scintillatorThickness):
    distance_to_end = scint_thickness_to_end(z, nLayersNS, substrateThickness, dielectricThickness, scintillatorThickness)
    mu = k_to_absorption_coeff(k)
    res = np.exp(-distance_to_end * mu)
    return res

def does_photon_count(z, k, nLayersNS, substrateThickness, dielectricThickness, scintillatorThickness):
    stay_end = abs_to_the_end(z, k, nLayersNS, substrateThickness, dielectricThickness, scintillatorThickness)
    rand = random.uniform(0, 1)
    if rand <= stay_end:
        return True
    return False

def simulate_sl_thickness(nLayersNS, substrateThickness, scintillatorThickness, slThickness, lightAbsorption, numEvents):
    print('     sl thickness:', int(slThickness*1e6), 'nm')
    scenario = simulation_template.format(worldSizeXY=1., worldSizeZ=1., nLayersNS=nLayersNS, substrateThickness=substrateThickness, scintillatorThickness=scintillatorThickness, dielectricThickness=slThickness, numEvents=numEvents)
    photons_Z = simulate_scenario_field(scenario, property='fCreatorProcess', key='Scintillation', field='fZ')
    does_photon_count_cur = lambda z: does_photon_count(z, lightAbsorption, nLayersNS, substrateThickness, slThickness, scintillatorThickness)
    num_photons_Z_reaching_end = len([pz for pz in photons_Z if does_photon_count_cur(pz)])
    return num_photons_Z_reaching_end

def ml_hybrid_simulation_as_a_function_of_sl_thickness(pairs, substrateThickness, scintillatorThicknesses, slThicknesses, lightAbsorption, numEvents, simulation_file):
    global simulate_sl_t # ugly trick to make the function pickable to pass it to the parallel for loop
    scint_sl_photons = []
    for k in pairs:
        nLayersNS = 2*k + 1
        print('nLayersNS:', int(nLayersNS))
        scint_sl_photons_tmp = []
        for scintillatorThickness in scintillatorThicknesses:
            print('  scintillator thickness:', int(scintillatorThickness*1e6), 'nm')
            def simulate_sl_t(sl_t): 
                return simulate_sl_thickness(nLayersNS, substrateThickness, scintillatorThickness, sl_t, lightAbsorption, numEvents)

            scintillatingPhotons = parallel_for_loop(simulate_sl_t, slThicknesses)
            scint_sl_photons_tmp.append(scintillatingPhotons)
        scint_sl_photons.append(scint_sl_photons_tmp)
    np.save(simulation_file, scint_sl_photons)
    return scint_sl_photons

def main():
    # Constants
    pairs = [1]
    substrateThickness = 0.0001e-3
    scintillatorThicknesses = [1e-3]
    slThicknesses = np.logspace(-5, -1, 3*multiprocessing.cpu_count() - 1)
    lightAbsorption = 0.015
    numEvents = int(3000000)

    mu_Ti = 1.107e2
    mu_Cl = 5.725e1
    mu_C  = 2.373
    mu_O  = 5.952
    TiFractions = 0.3923
    OFractions = 0.4066
    ClFractions = 0.0719
    CFractions = 0.1290
    density_sl = 1.3 + TiFractions/0.4 #g/cm^3
    mu_gamma_sl = density_sl * 1e-4 * (mu_Ti * TiFractions + mu_O * OFractions + mu_Cl * ClFractions + mu_C * CFractions)
    mu_gamma_scint = 1e-4 * 1.02 * (2.562e-1)
    mu_electron = 38.95632914365327 * 1e-3
    mu_electron_scint = 25.77332803210813 * 1e-3
    mu_sl_l = k_to_absorption_coeff(lightAbsorption)*1e-3
    mu_l_scint = 0.
    C_sl = 1.
    C_scint = 1.

    # Generate simulation results
    simulation_file = '/home/vboxuser/Projects/G4NS/G4_Nanophotonic_Scintillator/build/sim.npy'
    generate_simulation_results = True
    if generate_simulation_results:
        ml_hybrid_simulation_as_a_function_of_sl_thickness(pairs, substrateThickness, scintillatorThicknesses, slThicknesses, lightAbsorption, numEvents, simulation_file)
    simulation_photons = np.load(simulation_file)

    def theory_N(scintillatorThickness, slThickness, k):
        theoryPhotons = N_l_total(torch.Tensor(k * [scintillatorThickness]), torch.Tensor(k * [slThickness]), mu_gamma_scint, mu_electron_scint, mu_l_scint, mu_gamma_sl, mu_electron, mu_sl_l, C_sl, C_scint)
        return theoryPhotons

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    i = 0
    for k_ind in range(len(pairs)):
        k = pairs[k_ind]
        n_layers = 2*k+1
        scint_ind = 0
        for scint_ind in range(len(scintillatorThicknesses)):
            scintillatorThickness = scintillatorThicknesses[scint_ind]
            sim = simulation_photons[k_ind][scint_ind]
            sim = scipy.signal.savgol_filter(sim, window_length=10, polyorder=2)
    
            sim /= sim[0]
            ax.plot(slThicknesses*1e3, sim, label='n_layers=' + str(n_layers) + ', scint layer=' + str(int(scintillatorThickness*1e3)) + ' um')

            theory_photons_N = np.array([theory_N(scintillatorThickness*1e3, sl_t*1e3, k) for sl_t in slThicknesses])
            theory_photons_N /= theory_photons_N[0]
            ax.plot(slThicknesses*1e3, theory_photons_N, label='theory')


    ax.set_xlabel('SL thickness [um]')
    ax.set_ylabel('relative photons count')
    ax.set_xscale("log")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()