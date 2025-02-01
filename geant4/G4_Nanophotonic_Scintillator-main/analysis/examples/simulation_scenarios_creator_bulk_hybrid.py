import os, sys
sys.path.insert(0, '../')
import analysis.read_simulation as rs
# Run the script from the build directory

simulation_template = \
"""
/structure/isMultilayerNS 1
/structure/nLayersNS {nLayersNS}
/structure/substrateThickness {substrateThickness}
/structure/scintillatorThickness {scintillatorThickness}
/structure/dielectricThickness {dielectricThickness}

/structure/constructDetectors 0
/structure/nGridX 10
/structure/nGridY 10
/structure/detectorDepth 0.1e-3

/structure/checkDetectorsOverlaps 0

/run/initialize

/run/beamOn {numEvents}
"""

nLayersNS = 2
substrateThickness = 0.1e-3
dielectricThickness = 0
numEvents = 100000

output_location = 'run.mac'
root_file_name = 'output0.root'
scintillatorThicknesses = [2.5e-3, 3.1e-3, 3.9e-3, 5.0e-3, 6.4e-3]
scintillatingPhotons = []
for t in scintillatorThicknesses:
    with open(output_location, 'w') as f:
        f.writelines(simulation_template.format(nLayersNS=nLayersNS, substrateThickness=substrateThickness, scintillatorThickness=t, dielectricThickness=dielectricThickness, numEvents=numEvents))
    os.system('./NS ' + output_location)
    photonsX, photonsY, photonsZ = rs.read_simulation(root_file_name, property='fCreatorProcess', key='Scintillation')
    scintillatingPhotons.append(len(photonsZ))

print(scintillatorThicknesses)
print(scintillatingPhotons)
print([2.1*(n / scintillatingPhotons[-1]) for n in scintillatingPhotons])

