import pandas as pd
import uproot
import numpy as np
from scipy.stats import gaussian_kde
import multiprocessing
from mayavi import mlab

# reading root file
root_file = uproot.open('../build/output0.root')
photons = root_file["Photons"]

# Considering only optical photons
photonType = [t for t in photons["fType"].array()]
photonZ_all = np.array(photons["fZ"].array())
relevant_ind = [t == 'opticalphoton' and z >= 0 for t, z in zip(photonType, photonZ_all)] 
photonsX = np.array(photons["fX"].array())[relevant_ind]
photonsY = np.array(photons["fY"].array())[relevant_ind]
photonsZ = np.array(photons["fZ"].array())[relevant_ind]

def calc_kde(data):
    return kde(data.T)

# Calculate kernel density estimation
kde = gaussian_kde(np.row_stack((photonsX, photonsY, photonsZ)))

# Evaluate kde on a grid
factor = 1
factorZ = 1
grid_size = 30j
xmin, ymin, zmin = photonsX.min()/factor, photonsY.min()/factor, photonsZ.min()
xmax, ymax, zmax = photonsX.max()/factor, photonsY.max()/factor, photonsZ.max()
xi, yi, zi = np.mgrid[xmin:xmax:grid_size, ymin:ymax:grid_size, zmin:zmax:grid_size]
coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 

# Multiprocessing
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
results = pool.map(calc_kde, np.array_split(coords.T, 2))
density = np.concatenate(results).reshape(xi.shape)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot', bgcolor=(1,1,1), fgcolor=(0,0,0))

grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
mlab.pipeline.volume(grid, vmin=density.min() + .2*(density.max()-density.min()), vmax=density.min() + .5*(density.max()-density.min()))


mlab.axes()
mlab.show()