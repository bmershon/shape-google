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

n = 10
radius = 5
m = PolyMesh()
m.loadFile(sys.argv[1]) #Load a mesh
(Ps, Ns) = samplePointCloud(m, 20000) #Sample 20,000 points and associated normals
hist = getShapeHistogram(Ps, Ns, n, radius)

plt.stem(np.linspace(0, radius, n), hist)
plt.xlabel('Distance')
plt.ylabel('Probability')
plt.title('Shape Shell Histogram')

plt.show()