import sys
import os
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(here + "/../S3DGLPy")
sys.path.append(here + "/../")
from Primitives3D import *
from PolyMesh import *
import numpy as np
import matplotlib.pyplot as plt

from ShapeStatistics import *

np.random.seed(100) #Replace 100 with some number you both agree on

resolution = 2

m = PolyMesh()
m.loadOffFileExternal(sys.argv[1]) #Load a mesh
(Ps, Ns) = samplePointCloud(m, 20000) #Sample 20,000 points and associated normals
sphere = getSphereSamples(resolution)

bins = np.linspace(0, radius, sphere.shape[1] + 1)
hist = getShapeShellHistogram(Ps, Ns, n, radius, sphere)
plt.bar(bins[:-1], hist, width=bins[1]-bins[0])

plt.xlabel('Shell radius (with sectors)')
plt.ylabel('Frequency')
plt.title("Sectored Shell Histogram")

plt.show()