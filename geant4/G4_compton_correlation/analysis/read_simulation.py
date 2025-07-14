import pandas as pd
import uproot
import numpy as np

def read_simulation(root_file_name, property, key):
    # reading root file
    root_file = uproot.open(root_file_name)
    photons = root_file["Photons"]

    prop = [t for t in photons[property].array()]
    relevant_ind = [t == key for t in prop]
    photonsX = np.array(photons["fX"].array())[relevant_ind]
    photonsY = np.array(photons["fY"].array())[relevant_ind]
    photonsZ = np.array(photons["fZ"].array())[relevant_ind]
    
    return photonsX, photonsY, photonsZ

def read_simulation_field(root_file_name, property, key, field):
    # reading root file
    root_file = uproot.open(root_file_name)
    try:
        photons = root_file["Photons"]
    except:
        print('Error: empyt root file ' + root_file_name)
        return None

    prop = [t for t in photons[property].array()]
    relevant_ind = [t == key for t in prop]
    photonsField = np.array(photons[field].array())[relevant_ind]
    return photonsField