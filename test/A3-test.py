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

N = 5000
n = 20

m = PolyMesh()
m.loadFile(sys.argv[1]) #Load a mesh
(Ps, Ns) = samplePointCloud(m, 20000) #Sample 20,000 points and associated normals
hist = getA3Histogram(Ps, Ns, n, N)

print hist