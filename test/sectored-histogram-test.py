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

resolution = 1
n = 10
radius = 3
m = PolyMesh()
m.loadFile(sys.argv[1]) #Load a mesh
(Ps, Ns) = samplePointCloud(m, 20000) #Sample 20,000 points and associated normals
sphere = getSphereSamples(resolution)
hist = getShapeShellHistogram(Ps, Ns, n, radius, sphere)
plt.plot(hist)
plt.xlabel('Ordered Sectors (increasing radius, decreasing sector count)')
plt.ylabel('Frequency')
plt.title('Sectored Shape Shell Histogram')

plt.show()