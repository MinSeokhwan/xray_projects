import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-5])

import numpy as np
import util.read_txt as txt

target_material = 'YAG'
E = 116 #keV
density = 4.57 #g/cm^3
thickness = np.linspace(0.01, 0.1, 100) #cm

CS_raw = txt.read_txt(directory[:-5] + '/material_data/ComptCS_' + target_material)
CS_Compt = 10**np.interp(np.log10(E), np.log10(CS_raw[:,0]), np.log10(CS_raw[:,1]))

CS_raw = txt.read_txt(directory[:-5] + '/material_data/PhotCS_' + target_material)
CS_Phot = 10**np.interp(np.log10(E), np.log10(CS_raw[:,0]), np.log10(CS_raw[:,1]))

CS_tot = CS_Compt + CS_Phot
mfp = 1/(CS_tot*density)

P_trans = np.exp(-thickness/mfp)
P_Phot = (1 - P_trans)*CS_Phot/CS_tot

P_Compt1 = (1 - P_trans - P_Phot)*(thickness/mfp)*np.exp(-thickness/mfp)
P_ComptN = 1 - P_trans - P_Phot - P_Compt1

np.savez(directory + '/approx_interaction_probability', thickness=thickness, P_trans=P_trans, P_Phot=P_Phot, P_Compt1=P_Compt1, P_ComptN=P_ComptN)