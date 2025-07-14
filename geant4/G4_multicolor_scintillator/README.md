# G4_Nanophotonic_Scintillator

A simulation tool based on Geant4 software that allows to simulate and analyze the scintillation from nanophotonic scintillators.
The tool allows to control the geometry, particle gun, detectors, materials, and other simulation parameters.

## Installation
The tool relies on an external installation of Geant4. Please visit https://geant4.web.cern.ch/download/ for more information.
Once installing the Geant4 software, clone this repo.

This is a c++ and CMake project. To execute the project, create a build folder.
In the build folder configure the project by executing "cmake .." and build it with "make".

## Simulation
Determining the simulation paramerters can be done in two ways.
The first approach is to modify construction.cc, by coding the required changes (for example, geometry and materials). This require re-compiling the project.
Another approach, is to use currrent configuration with other parameters, by modifying the mac/run.mac file.

## Data analysis
The analysis of the simulation results is done in Python scripts in the analysis folder.
The scripts allows to consider only the interesting particle and physical processes from the simulation, and visualize them in different ways.

For additional help, please contact Avner Shultzman - avners8@gmail.com.
