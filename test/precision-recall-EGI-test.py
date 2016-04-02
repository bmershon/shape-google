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

NUM_PER_CLASS = 10
POINTCLOUD_CLASSES = ['biplane', 'desk_chair', 'dining_chair', 'fighter_jet', 'fish', 'flying_bird', 'guitar', 'handgun', 'head', 'helicopter', 'human', 'human_arms_out', 'potted_plant', 'race_car', 'sedan', 'shelves', 'ship', 'sword', 'table', 'vase']

NRandSamples = 10000 #You can tweak this number
np.random.seed(42) #For repeatable results randomly sampling
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
# Precision recall for all classes of shapes, averaged together
H0 = shp.makeAllHistograms(PointClouds, Normals, shp.getEGIHistogram, shp.getSphereSamples(1))
H1 = shp.makeAllHistograms(PointClouds, Normals, shp.getEGIHistogram, shp.getSphereSamples(2))
H2 = shp.makeAllHistograms(PointClouds, Normals, shp.getEGIHistogram, shp.getSphereSamples(3))
 
D0 = shp.compareHistsEuclidean(H0)
D1 = shp.compareHistsEuclidean(H1)
D2 = shp.compareHistsEuclidean(H2)
 
PR0 = shp.getPrecisionRecall(D0)
PR1 = shp.getPrecisionRecall(D1)
PR2 = shp.getPrecisionRecall(D2)
 
recalls = np.linspace(1.0/9.0, 1.0, 9)
plt.hold(True)
plt.plot(recalls, PR0, 'r', label='EGI (18 directions)')
plt.plot(recalls, PR1, 'k', label='EGI (66 directions)')
plt.plot(recalls, PR2, 'c', label='EGI (258 directions)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig(sys.argv[1])