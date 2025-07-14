import os, sys
home_dir = '/home/vboxuser/Projects/G4NS/G4_Nanophotonic_Scintillator/'
sys.path.insert(0, home_dir)
import analysis.read_simulation as rs
# Run the script from the build directory
import random
import numpy as np
import analysis.g2_analysis.g2 as g2a
from utils.utils import *

"""
This script simulate the many events of single emitted x-ray/electron.
If is_yeild is true, the script will iterate over multiple scintillation yeild.
Otherwise, the script will iterate over multiple scintillation lifetimes.
In any case, for each iteration (a given yeild or lifetime), we will create a new folder and store in it the results of all the single excitation events.
"""


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
"""

substrateThickness = 0.01e-3
scintillatorThickness = 10e-3
numEvents = 10000
numPhotonsPerElectron = 100
electronEnergy = 10. # in keV
is_yield = False

if is_yield:
    scintillatorLifeTime = 2.5
    scintillationYield_array = np.logspace(2, 4, 2*multiprocessing.cpu_count() - 1)
    scint_array = scintillationYield_array
else:
    scintillationYield = numPhotonsPerElectron / electronEnergy * 1000.
    scintillatorLifeTime_array = np.linspace(1., 10., 10)
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
                    scintillatorLifeTime=t, scintillationYield=scintillationYield)
    simulate_scenario_field_multiple(scenario, nEvents=numEvents, i=t)
    return 0

for t in scint_array:
    generate_and_simulate(t)
