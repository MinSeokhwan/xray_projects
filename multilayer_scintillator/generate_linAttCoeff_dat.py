import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
from scipy.io import loadmat
import util.read_txt as txt

def linAttCoeff_dat(filename, energy_list, mediaIndex, mediaName):
    media = txt.read_txt(directory + "/data/ICRP110/" + filename + "_media")/100
    density = txt.read_txt(directory + "/data/ICRP110/elemental_density")
    density_media = txt.read_txt(directory + "/data/ICRP110/media_density")

    energy_bins = energy_list.size
    N_media = media.shape[0]
    N_elements = media.shape[1]
    
    element_list = ['H','C','N','O','Na','Mg','P','S','Cl','K','Ca','Fe','I','Gd']
    massAttCoeff_element = np.zeros((energy_bins, N_elements))
    for ne in range(N_elements):
        raw = txt.read_txt(directory + '/data/linAttCoeff' + element_list[ne])
        massAttCoeff_element[:,ne] = np.interp(energy_list, raw[:,0], raw[:,1], left=0, right=0)/density[ne]
    
    linAttCoeff_media = density_media[:-2,np.newaxis]*(media @ massAttCoeff_element.T)
    
#    if filename == 'AM':
#        media_list = ['Teeth','MineralBone','HumeriUpper','HumeriLower','LowerArmBone','HandBone','Clavicles','Cranium','FemoraUpper','FemoraLower','LowerLeg','Foot','Mandible','Pelvis','Ribs',
#                      'Scapulae','CervicalSpine','ThoracicSpine','LumbarSpine','Sacrum','Sternum','HumeriFemoraUpperMedullaryCavity','HumeriFemoraLowerMedullaryCavity','LowerArmMedullaryCavity',
#                      'LowerLegMedullaryCavity','Cartilage','Skin','Blood','Muscle','Liver','Pancreas','Brain','Heart','Eyes','Kidneys','Stomach','SmallIntestine','LargeIntestine','Spleen','Thyroid',
#                      'UrinaryBladder','Testes','Adrenals','Oesophagus','Gallbladder','Prostate','Lymph','Breast','AdiposeTissue','Lung','GastroIntestinalContents','Urine','Air','IodinatedBlood',
#                      'GadolinatedBlood']
#    elif filename == 'AF':
#        media_list = ['Teeth','MineralBone','HumeriUpper','HumeriLower','LowerArmBone','HandBone','Clavicles','Cranium','FemoraUpper','FemoraLower','LowerLeg','Foot','Mandible','Pelvis','Ribs',
#                      'Scapulae','CervicalSpine','ThoracicSpine','LumbarSpine','Sacrum','Sternum','HumeriFemoraUpperMedullaryCavity','HumeriFemoraLowerMedullaryCavity','LowerArmMedullaryCavity',
#                      'LowerLegMedullaryCavity','Cartilage','Skin','Blood','Muscle','Liver','Pancreas','Brain','Heart','Eyes','Kidneys','Stomach','SmallIntestine','LargeIntestine','Spleen','Thyroid',
#                      'UrinaryBladder','Ovaries','Adrenals','Oesophagus','Gallbladder','Uterus','Lymph','Breast','AdiposeTissue','Lung','GastroIntestinalContents','Urine','Air','IodinatedBlood',
#                      'GadolinatedBlood']
    
    with open("/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main/dat/linAttCoeff" + filename + mediaName + ".dat", 'w') as dat:
        for ne in range(energy_bins-1):
            dat.write('%.6f\t%.6f\n' %(energy_list[ne], linAttCoeff_media[mediaIndex,ne]))
        dat.write('%.6f\t%.6f' %(energy_list[-1], linAttCoeff_media[mediaIndex,-1]))

    return linAttCoeff_organs
    
def linAttCoeff_contrast_dat(filename, energy_list, conc, mediaName, G4directory):
    media = txt.read_txt(directory + "/data/ICRP110/" + filename + "_media")/100
    media_temp = media[27,:]*(1 - conc)
    if mediaName == 'IBlood':
        media_temp[-2] = conc
    elif mediaName == 'GdBlood':
        media_temp[-1] = conc
    media_temp = media_temp.reshape(1, -1)
    density = txt.read_txt(directory + "/data/ICRP110/elemental_density")
    density_media = txt.read_txt(directory + "/data/ICRP110/media_density")

    energy_bins = energy_list.size
    N_media = media.shape[0]
    N_elements = media.shape[1]
    
    element_list = ['H','C','N','O','Na','Mg','P','S','Cl','K','Ca','Fe','I','Gd']
    massAttCoeff_element = np.zeros((energy_bins, N_elements))
    for ne in range(N_elements):
        raw = txt.read_txt(directory + '/data/linAttCoeff' + element_list[ne])
        massAttCoeff_element[:,ne] = np.interp(energy_list, raw[:,0], raw[:,1], left=0, right=0)/density[ne]
    
    linAttCoeff_media = density_media*(media_temp @ massAttCoeff_element.T)
    
#    if filename == 'AM':
#        media_list = ['Teeth','MineralBone','HumeriUpper','HumeriLower','LowerArmBone','HandBone','Clavicles','Cranium','FemoraUpper','FemoraLower','LowerLeg','Foot','Mandible','Pelvis','Ribs',
#                      'Scapulae','CervicalSpine','ThoracicSpine','LumbarSpine','Sacrum','Sternum','HumeriFemoraUpperMedullaryCavity','HumeriFemoraLowerMedullaryCavity','LowerArmMedullaryCavity',
#                      'LowerLegMedullaryCavity','Cartilage','Skin','Blood','Muscle','Liver','Pancreas','Brain','Heart','Eyes','Kidneys','Stomach','SmallIntestine','LargeIntestine','Spleen','Thyroid',
#                      'UrinaryBladder','Testes','Adrenals','Oesophagus','Gallbladder','Prostate','Lymph','Breast','AdiposeTissue','Lung','GastroIntestinalContents','Urine','Air','IodinatedBlood',
#                      'GadolinatedBlood']
#    elif filename == 'AF':
#        media_list = ['Teeth','MineralBone','HumeriUpper','HumeriLower','LowerArmBone','HandBone','Clavicles','Cranium','FemoraUpper','FemoraLower','LowerLeg','Foot','Mandible','Pelvis','Ribs',
#                      'Scapulae','CervicalSpine','ThoracicSpine','LumbarSpine','Sacrum','Sternum','HumeriFemoraUpperMedullaryCavity','HumeriFemoraLowerMedullaryCavity','LowerArmMedullaryCavity',
#                      'LowerLegMedullaryCavity','Cartilage','Skin','Blood','Muscle','Liver','Pancreas','Brain','Heart','Eyes','Kidneys','Stomach','SmallIntestine','LargeIntestine','Spleen','Thyroid',
#                      'UrinaryBladder','Ovaries','Adrenals','Oesophagus','Gallbladder','Uterus','Lymph','Breast','AdiposeTissue','Lung','GastroIntestinalContents','Urine','Air','IodinatedBlood',
#                      'GadolinatedBlood']
    
    with open(G4directory + "/dat/linAttCoeff" + filename + mediaName + ".dat", 'w') as dat:
        for ne in range(energy_bins-1):
            dat.write('%.6f\t%.6f\n' %(energy_list[ne], linAttCoeff_media[0,ne]))
        dat.write('%.6f\t%.6f' %(energy_list[-1], linAttCoeff_media[0,-1]))

def linAttCoeff(filename, energy_list, IConc, GdConc, G4directory):
    media = txt.read_txt(directory + "/data/ICRP110/" + filename + "_media")/100
    mediaIBlood = media[27,:]*(1 - IConc)
    mediaIBlood[-2] = IConc
    mediaGdBlood = media[27,:]*(1 - GdConc)
    mediaGdBlood[-1] = GdConc
    media = np.concatenate((media, mediaIBlood.reshape(1,-1), mediaGdBlood.reshape(1,-1)), axis=0)
    density = txt.read_txt(directory + "/data/ICRP110/elemental_density")
    density_media = txt.read_txt(directory + "/data/ICRP110/media_density")

    energy_bins = energy_list.size
    N_media = media.shape[0]
    N_elements = media.shape[1]
    
    element_list = ['H','C','N','O','Na','Mg','P','S','Cl','K','Ca','Fe','I','Gd']
    massAttCoeff_element = np.zeros((energy_bins, N_elements))
    for ne in range(N_elements):
        raw = txt.read_txt(directory + '/data/linAttCoeff' + element_list[ne])
        temp = np.interp(np.log10(energy_log), np.log10(raw[:,0]), np.log10(raw[:,1]), left=0, right=0)
        massAttCoeff_element[:,ne] = 10**temp/density[ne]
    
    linAttCoeff_media = density_media*(media @ massAttCoeff_element.T)
    
    with open(G4directory + "/dat/linAttCoeff" + filename + "IBlood.dat", 'w') as dat:
        for ne in range(energy_bins-1):
            dat.write('%.6f\t%.6f\n' %(energy_list[ne], linAttCoeff_media[-2,ne]))
        dat.write('%.6f\t%.6f' %(energy_list[-1], linAttCoeff_media[-2,-1]))
    
    with open(G4directory + "/build/linAttCoeff" + filename + "IBlood.dat", 'w') as dat:
        for ne in range(energy_bins-1):
            dat.write('%.6f\t%.6f\n' %(energy_list[ne], linAttCoeff_media[-2,ne]))
        dat.write('%.6f\t%.6f' %(energy_list[-1], linAttCoeff_media[-2,-1]))
        
    with open(G4directory + "/dat/linAttCoeff" + filename + "GdBlood.dat", 'w') as dat:
        for ne in range(energy_bins-1):
            dat.write('%.6f\t%.6f\n' %(energy_list[ne], linAttCoeff_media[-1,ne]))
        dat.write('%.6f\t%.6f' %(energy_list[-1], linAttCoeff_media[-1,-1]))
    
    with open(G4directory + "/build/linAttCoeff" + filename + "GdBlood.dat", 'w') as dat:
        for ne in range(energy_bins-1):
            dat.write('%.6f\t%.6f\n' %(energy_list[ne], linAttCoeff_media[-1,ne]))
        dat.write('%.6f\t%.6f' %(energy_list[-1], linAttCoeff_media[-1,-1]))
    
    organs = txt.read_txt(directory + "/data/ICRP110/" + filename + "_organs")
    N_organs = organs.shape[0] + 1
    
    linAttCoeff_organs = np.zeros((N_organs, energy_bins))
    for no in range(1, N_organs):
        linAttCoeff_organs[no,:] = linAttCoeff_media[int(organs[no-1,0]-1),:]
    linAttCoeff_organs[0,:] = linAttCoeff_organs[-1,:]
    
#    if filename == 'AM':
#        media_list = ['Teeth','MineralBone','HumeriUpper','HumeriLower','LowerArmBone','HandBone','Clavicles','Cranium','FemoraUpper','FemoraLower','LowerLeg','Foot','Mandible','Pelvis','Ribs',
#                      'Scapulae','CervicalSpine','ThoracicSpine','LumbarSpine','Sacrum','Sternum','HumeriFemoraUpperMedullaryCavity','HumeriFemoraLowerMedullaryCavity','LowerArmMedullaryCavity',
#                      'LowerLegMedullaryCavity','Cartilage','Skin','Blood','Muscle','Liver','Pancreas','Brain','Heart','Eyes','Kidneys','Stomach','SmallIntestine','LargeIntestine','Spleen','Thyroid',
#                      'UrinaryBladder','Testes','Adrenals','Oesophagus','Gallbladder','Prostate','Lymph','Breast','AdiposeTissue','Lung','GastroIntestinalContents','Urine','Air','IodinatedBlood',
#                      'GadolinatedBlood']
#    elif filename == 'AF':
#        media_list = ['Teeth','MineralBone','HumeriUpper','HumeriLower','LowerArmBone','HandBone','Clavicles','Cranium','FemoraUpper','FemoraLower','LowerLeg','Foot','Mandible','Pelvis','Ribs',
#                      'Scapulae','CervicalSpine','ThoracicSpine','LumbarSpine','Sacrum','Sternum','HumeriFemoraUpperMedullaryCavity','HumeriFemoraLowerMedullaryCavity','LowerArmMedullaryCavity',
#                      'LowerLegMedullaryCavity','Cartilage','Skin','Blood','Muscle','Liver','Pancreas','Brain','Heart','Eyes','Kidneys','Stomach','SmallIntestine','LargeIntestine','Spleen','Thyroid',
#                      'UrinaryBladder','Ovaries','Adrenals','Oesophagus','Gallbladder','Uterus','Lymph','Breast','AdiposeTissue','Lung','GastroIntestinalContents','Urine','Air','IodinatedBlood',
#                      'GadolinatedBlood']
    
#    for nm in range(N_media):
#        with open("/home/gridsan/smin/geant4/G4_Nanophotonic_Scintillator-main/dat/linAttCoeff" + filename + media_list[nm] + ".dat", 'w') as dat:
#            for ne in range(energy_bins-1):
#                dat.write('%.6f\t%.6f\n' %(energy_list[ne], linAttCoeff_media[nm,ne]))
#            dat.write('%.6f\t%.6f' %(energy_list[-1], linAttCoeff_media[nm,-1]))

    return linAttCoeff_organs
    
def linAttCoeff_cylinder_phantom(filename, energy_list, IConc_list, GdConc_list, media_index): #1 Mineral bone, 28 Muscle tissue, 48 Adipose tissue
    media_temp = txt.read_txt(directory + "/data/ICRP110/" + filename + "_media")/100
    media = media_temp[media_index,:]
    density = txt.read_txt(directory + "/data/ICRP110/elemental_density")
    density_media = txt.read_txt(directory + "/data/ICRP110/media_density")
    media_index = np.append(media_index, np.array([27]*(IConc_list.size + GdConc_list.size)))
    density_media = density_media[media_index]
    
    for ni in range(IConc_list.size):
        mediaIBlood = media_temp[27,:]*(1 - IConc_list[ni])
        mediaIBlood[-2] = IConc_list[ni]
        media = np.concatenate((media, mediaIBlood.reshape(1,-1)), axis=0)
    
    for ng in range(GdConc_list.size):
        mediaGdBlood = media_temp[27,:]*(1 - GdConc_list[ng])
        mediaGdBlood[-1] = GdConc_list[ng]
        media = np.concatenate((media, mediaGdBlood.reshape(1,-1)), axis=0)

    energy_bins = energy_list.size
    N_media = media.shape[0]
    N_elements = media.shape[1]
    
    element_list = ['H','C','N','O','Na','Mg','P','S','Cl','K','Ca','Fe','I','Gd']
    massAttCoeff_element = np.zeros((energy_bins, N_elements))
    for ne in range(N_elements):
        raw = txt.read_txt(directory + '/data/linAttCoeff' + element_list[ne])
        temp = np.interp(np.log10(energy_list), np.log10(raw[:,0]), np.log10(raw[:,1]), left=0, right=0)
        massAttCoeff_element[:,ne] = 10**temp/density[ne]
    
    linAttCoeff_media = density_media*(media @ massAttCoeff_element.T)

    return linAttCoeff_media
    
def linAttCoeff_cylinder_phantom_battery(energy_list):
    density = txt.read_txt(directory + "/data/elemental_density_battery")
    density_media = txt.read_txt(directory + "/data/media_density_battery")
    media = np.array([[0    ,0  ,0    ,1,0,0    ,0,0    ,0    ,0,0,0], #Al
                      [0.033,0  ,0.151,0,0,0.260,0,0.279,0.277,0,0,0], #NMC
                      [0    ,0  ,0    ,0,0,0    ,0,0    ,0    ,1,0,0], #Cu
                      [0    ,0  ,0    ,0,0,0    ,0,0    ,1    ,0,0,0], #Ni
                      [0    ,0  ,0    ,0,0,0    ,0,0    ,0    ,0,0,1], #Pb
                      [0    ,0  ,0    ,0,0,0    ,1,0    ,0    ,0,0,0], #Fe
                      [0    ,0  ,0    ,0,1,0    ,0,0    ,0    ,0,0,0], #Cr
                      [0    ,0  ,0    ,0,0,0    ,0,0    ,0    ,0,1,0], #Zn
                      [0    ,0.8,0.2  ,0,0,0    ,0,0    ,0    ,0,0,0]])#Air
    
    energy_bins = energy_list.size
    N_elements = media.shape[1]
    
    element_list = ['Li','N','O','Al','Cr','Mn','Fe','Co','Ni','Cu','Zn','Pb']
    massAttCoeff_element = np.zeros((energy_bins, N_elements))
    for ne in range(N_elements):
        raw = txt.read_txt(directory + '/data/linAttCoeff' + element_list[ne])
        temp = np.interp(np.log10(energy_list), np.log10(raw[:,0]), np.log10(raw[:,1]), left=0, right=0)
        massAttCoeff_element[:,ne] = 10**temp/density[ne]
    
    linAttCoeff_media = density_media*(media @ massAttCoeff_element.T)

    return linAttCoeff_media
    
def linAttCoeff_cylinder_phantom_baggage(energy_list):
    density = txt.read_txt(directory + "/data/elemental_density_baggage")
    density_media = txt.read_txt(directory + "/data/media_density_baggage")
    media = np.array([[0    ,0    ,0.8  ,0.2  ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0,0], #Air
                      [0.042,0.625,0    ,0.333,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0,0], #PET
                      [0.144,0.856,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0,0], #HDPE
                      [0.102,0.142,0.034,0.711,0.001,0    ,0    ,0.002,0.003,0.001,0.004,0,0], #muscle
                      [0.112,0    ,0    ,0.888,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0,0], #H2O
                      [0.062,0.445,0    ,0.493,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0,0], #cellulose
                      [0    ,0    ,0    ,0.533,0    ,0    ,0.467,0    ,0    ,0    ,0    ,0,0], #SiO2
                      [0    ,0    ,0    ,0.535,0    ,0.127,0.338,0    ,0    ,0    ,0    ,0,0], #ceramic
                      [0    ,0    ,0    ,0    ,0    ,1    ,0    ,0    ,0    ,0    ,0    ,0,0], #Al
                      [0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,1,0], #Fe
                      [0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0    ,0,1]])#Au
    
    energy_bins = energy_list.size
    N_elements = media.shape[1]
    
    element_list = ['H','C','N','O','Na','Al','Si','P','S','Cl','K','Fe','Au']
    massAttCoeff_element = np.zeros((energy_bins, N_elements))
    for ne in range(N_elements):
        raw = txt.read_txt(directory + '/data/linAttCoeff' + element_list[ne])
        temp = np.interp(np.log10(energy_list), np.log10(raw[:,0]), np.log10(raw[:,1]), left=0, right=0)
        massAttCoeff_element[:,ne] = 10**temp/density[ne]
    
    linAttCoeff_media = density_media*(media @ massAttCoeff_element.T)

    return linAttCoeff_media

def phantom(filename, dimZ, indPhantom, energy_list, energy_tgt_ind, xraySrc, IConc, GdConc, G4directory):
    phantom = loadmat(directory + "/data/ICRP110/" + filename + "_3D.mat")['phantom'].transpose(0,2,1)
    phantom = phantom[indPhantom[0,0]:indPhantom[0,1],indPhantom[1,0]:indPhantom[1,1],indPhantom[2,0]:indPhantom[2,1]]
    mu = linAttCoeff(filename, energy_list, IConc, GdConc, G4directory)
    
    Nx = int(indPhantom[0,1] - indPhantom[0,0])
    Ny = int(indPhantom[1,1] - indPhantom[1,0])
    Nz = int(indPhantom[2,1] - indPhantom[2,0])
    
    images = np.ones((Nx, Ny, energy_list.size))*xraySrc[np.newaxis,np.newaxis,:]
    
    for nx in range(Nx):
        for ny in range(Ny):
            for nz in range(Nz):
                images[nx,ny,:] *= np.exp(-mu[phantom[nx,ny,nz],:]*dimZ)
    
    np.savez(directory + "/data/linAttCoeff_organs", mu=mu, phantom=phantom, images=images)
    
    images_by_interval = np.zeros((Nx, Ny, energy_tgt_ind.size-1))
    for ne in range(energy_tgt_ind.size-1):
        images_by_interval[:,:,ne] = np.sum(images[:,:,energy_tgt_ind[ne]:energy_tgt_ind[ne+1]], axis=2)/(Nx*Ny*np.sum(xraySrc[energy_tgt_ind[ne]:energy_tgt_ind[ne+1]]))
    
    return images_by_interval, phantom

def phantom_CT(filename, ind_slice, energy_list, xraySrc, IConc, GdConc, G4directory):
    phantom = loadmat(directory + "/data/ICRP110/" + filename + "_3D.mat")['phantom'].transpose(0,2,1)
    phantom = phantom[:,ind_slice,:]
    
    N0, N1 = phantom.shape
    lenDiag = int(np.ceil(np.sqrt(N0**2+N1**2)))
    if lenDiag%2 == 1:
        lenDiag += 1
    if N0%2 == 1:
        c0bef = int(np.floor((lenDiag - N0)/2))
        c0aft = c0bef + 1
    else:
        c0bef = int(np.round((lenDiag - N0)/2))
        c0aft = c0bef
    if N1%2 == 1:
        c1bef = int(np.floor((lenDiag - N1)/2))
        c1aft = c1bef + 1
    else:
        c1bef = int(np.round((lenDiag - N1)/2))
        c1aft = c1bef
    phantom_pad = np.pad(phantom, pad_width=((c0bef,c0aft),(c1bef,c1aft)), mode='edge')
    
    mu = linAttCoeff(filename, energy_list, IConc, GdConc, G4directory)

    Nx, Nz = phantom_pad.shape
    mu_mapping = np.ones((Nx, Nz, energy_list.size))*xraySrc[np.newaxis,np.newaxis,:]
    
    for nx in range(Nx):
        for nz in range(Nz):
            mu_mapping[nx,nz,:] = mu[phantom_pad[nx,nz],:]
    
    return mu_mapping, phantom_pad

if __name__ == '__main__':
    filename = 'AM'
    energy_list = np.linspace(1, 1000, 1000)
    linAttCoeff_dat(filename, energy_list)