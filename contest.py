import sys
import os
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(here + "/S3DGLPy")
sys.path.append(here + "/")
from Primitives3D import *
from PolyMesh import *
import numpy as np
import matplotlib.pyplot as plt

import ShapeStatistics as shp

NUM_PER_CLASS = 10
POINTCLOUD_CLASSES = ['biplane', 'desk_chair', 'dining_chair', 'fighter_jet', 'fish', 'flying_bird', 'guitar', 'handgun', 'head', 'helicopter', 'human', 'human_arms_out', 'potted_plant', 'race_car', 'sedan', 'shelves', 'ship', 'sword', 'table', 'vase']

NRandSamples = 10000 #You can tweak this number
np.random.seed(100) #For repeatable results randomly sampling
#Load in and sample all meshes
PointClouds = []
Normals = []
for i in range(len(POINTCLOUD_CLASSES)):
    print "LOADING CLASS %i of %i..."%(i, len(POINTCLOUD_CLASSES))
    PCClass = []
    for j in range(NUM_PER_CLASS):
        m = PolyMesh()
        filename = "models_off/%s%i.off"%(POINTCLOUD_CLASSES[i], j)
        print "Loading ", filename
        m.loadOffFileExternal(filename)
        (Ps, Ns) = shp.samplePointCloud(m, NRandSamples)
        PointClouds.append(Ps)
        Normals.append(Ns)


def getRandomHistogram(P, N, length):
    return np.random.randint(1000, size=length)

# Precision recall for all classes of shapes, averaged together
SPoints = shp.getSphereSamples(2)
HistsRandom = shp.makeAllHistograms(PointClouds, Normals, getRandomHistogram, 30)
 
D = shp.getMyShapeDistances(PointClouds, Normals)
DRandom = shp.compareHistsEuclidean(HistsRandom)
 
PR = shp.getPrecisionRecall(D)
PRRandom = shp.getPrecisionRecall(DRandom)
 
recalls = np.linspace(1.0/9.0, 1.0, 9)
plt.hold(True)
plt.plot(recalls, PR, 'k', label='Magic (More Magic)')
plt.plot(recalls, PRRandom, 'm', label='Random (Euclidean)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Magical Precision Recall Contest Entry')
plt.legend()
plt.savefig(sys.argv[1])