import os, sys
home_dir = '/home/vboxuser/Projects/G4NS/G4_Nanophotonic_Scintillator/'
sys.path.insert(0, home_dir)
import analysis.read_simulation as rs
# Run the script from the build directory
import random
import numpy as np
import analysis.g2_analysis.g2 as g2a
from utils.utils import *

simulation_template = \
"""
/structure/isMultilayerNS 1
/structure/nLayersNS 2
/structure/substrateThickness {substrateThickness}
/structure/scintillatorThickness {scintillatorThickness}
/structure/dielectricThickness 0

/structure/scintillatorLifeTime {scintillatorLifeTime}
/structure/scintillationYield {scintillationYield}

/run/initialize

/run/beamOn {numEvents}
"""

substrateThickness = 0.01e-3
scintillatorThickness = 1e-3
numEvents = 1000000

is_yield = True

if is_yield:
    scintillatorLifeTime = 2.5
    scintillationYield_array = np.logspace(2, 4, 1*multiprocessing.cpu_count() - 2)
    scint_array = scintillationYield_array
else:
    scintillationYield = 9000
    scintillatorLifeTime_array = np.linspace(1., 15., 1*multiprocessing.cpu_count())
    scint_array = scintillatorLifeTime_array

def generate_and_simulate(t):
    if is_yield:
        print('scintillation yield:', t, 'photons / MeV')
        scenario = simulation_template.format(
                    substrateThickness=substrateThickness, scintillatorThickness=scintillatorThickness,
                    scintillatorLifeTime=scintillatorLifeTime, scintillationYield=t, numEvents=numEvents)
    else:
        print('scintillation lifetime:', t, 'ns')
        scenario = simulation_template.format(
                    substrateThickness=substrateThickness, scintillatorThickness=scintillatorThickness,
                    scintillatorLifeTime=t, scintillationYield=scintillationYield, numEvents=numEvents)
    photonTime = simulate_scenario_field(scenario, property='fCreatorProcess', key='Scintillation', field='fT')
    if photonTime is None:
        return 0
    print('scintillation yield:', t, 'photons / MeV:::::',photonTime)
    time, g2 = g2a.compute_g2(photonTime)
    tau = fit_exp_param(time, g2)
    g2_0 = g2[0]
    print('scintillation yield:', t, 'photons / MeV:::::',g2)
    print('scintillation yield:', t, 'photons / MeV:::::',g2[0])
    return g2_0

fitted_g2 = np.array(parallel_for_loop(generate_and_simulate, scint_array))

print(scint_array.tolist())
print(fitted_g2.tolist())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(scint_array, fitted_g2)

if is_yield:
    ax.set_xscale("log")
ax.set_xlabel('scintillation yield [visible photons / MeV]')
ax.set_ylabel('g2(0)')
ax.set_title('g2(0) as a function of the scintillation yield')
plt.show()