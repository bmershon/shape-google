# command line: model, sphere output

import sys
import os
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(here + "/../S3DGLPy")
sys.path.append(here + "/../")
from Primitives3D import *
from PolyMesh import *
import numpy as np
import matplotlib.pyplot as plt

import ShapeStatistics as shp

np.random.seed(40) #Replace 100 with some number you both agree on

cmap = plt.get_cmap('jet') # color ramp
resolution = 2
n = 10

print sys.argv

m = PolyMesh()
m.loadFile(sys.argv[1]) #Load a mesh
(Ps, Ns) = shp.samplePointCloud(m, 20000) #Sample 20,000 points and associated normals
sphere = getSphereMesh(1, resolution)
hist = shp.getEGIHistogram(Ps, Ns, sphere.VPos.T)
hist = hist / np.max(hist)
sphere.VColors = np.array(np.round(255.0*cmap(hist)[:, 0:3]), dtype=np.int64)
sphere.saveOffFile(sys.argv[2], output255 = True)