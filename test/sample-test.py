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


m = PolyMesh()
m.loadOffFileExternal(sys.argv[1]) #Load a mesh
(Ps, Ns) = samplePointCloud(m, 20000) #Sample 20,000 points and associated normals
exportPointCloud(Ps, Ns, sys.argv[2]) #Export point cloud