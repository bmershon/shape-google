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

np.random.seed(100) #Replace 100 with some number you both agree on

n = 30

radius = 2
m = PolyMesh()
m.loadOffFileExternal(sys.argv[1]) #Load a mesh
(Ps, Ns) = shp.samplePointCloud(m, 20000) #Sample 20,000 points and associated normals

bins = np.linspace(0.0, radius, n + 1)
hist = shp.getShapeHistogram(Ps, Ns, n, radius)
plt.bar(bins[:-1], hist, width=bins[1]-bins[0])

plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title("Shell Histogram")

plt.show()
