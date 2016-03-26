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
m.loadFile(here + "/../models_off/biplane0.off") #Load a mesh
(Ps, Ns) = samplePointCloud(m, 20000) #Sample 20,000 points and associated normals
exportPointCloud(Ps, Ns, here + "/../build/biplane.pts") #Export point cloud