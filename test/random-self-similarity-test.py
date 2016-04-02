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

def randomHistogram(P, N, length):
    return np.random.randint(1000, size=length)

PointClouds = np.zeros(1000)
Normals = np.zeros(1000)

H = shp.makeAllHistograms(PointClouds, Normals, randomHistogram, 30)
D = shp.compareHistsEuclidean(H)

plt.imshow(D); plt.title("Random Histograms (Euclidean Distance)")
plt.savefig(sys.argv[1])      