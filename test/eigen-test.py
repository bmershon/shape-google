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

n = 10
radius = 2
m = PolyMesh()
m.loadOffFileExternal(sys.argv[1]) #Load a mesh
(Ps, Ns) = shp.samplePointCloud(m, 20000) #Sample 20,000 points and associated normals

bins = np.linspace(0, radius, 3*n + 1)
hist = shp.getShapeHistogramPCA(Ps, Ns, n, radius)
plt.bar(bins[:-1], hist, width=bins[1]-bins[0])

plt.xlabel('Eigenvalues # (increasing shell radius, decreasing value within shell)')
plt.ylabel('Frequency')
plt.title("Eigenvalue Shell Histogram")

plt.show()