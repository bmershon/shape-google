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

HistsShell = shp.makeAllHistograms(PointClouds, Normals, shp.getShapeHistogram, 30, 3.0)
HistsSector = shp.makeAllHistograms(PointClouds, Normals, shp.getShapeShellHistogram, 30, 3.0, SPoints)
HistsEGI = shp.makeAllHistograms(PointClouds, Normals, shp.getEGIHistogram, SPoints)
HistsA3 = shp.makeAllHistograms(PointClouds, Normals, shp.getA3Histogram, 30, 10000)
HistsD2 = shp.makeAllHistograms(PointClouds, Normals, shp.getD2Histogram, 3.0, 30, 10000)
HistsRandom = shp.makeAllHistograms(PointClouds, Normals, getRandomHistogram, 30)
 
DShell = shp.compareHistsEMD1D(HistsShell)
DSector = shp.compareHistsEMD1D(HistsSector)
DEGI = shp.compareHistsEMD1D(HistsEGI)
DA3 = shp.compareHistsEuclidean(HistsA3)
DD2 = shp.compareHistsEuclidean(HistsD2)
DRandom = shp.compareHistsEuclidean(HistsRandom)
 
PRShell = shp.getPrecisionRecall(DShell)
PRSector = shp.getPrecisionRecall(DSector)
PREGI = shp.getPrecisionRecall(DEGI)
PRA3 = shp.getPrecisionRecall(DA3)
PRD2 = shp.getPrecisionRecall(DD2)
PRRandom = shp.getPrecisionRecall(DRandom)
 
recalls = np.linspace(1.0/9.0, 1.0, 9)
plt.hold(True)
plt.plot(recalls, PRShell, 'k--', label='Shell (EMD)')
plt.plot(recalls, PRSector, 'r--', label='Shell + Sectors (EMD)')
plt.plot(recalls, PRA3, 'r', label='A3 (Euclidean)')
plt.plot(recalls, PRD2, 'k', label='D2 (Euclidean)')
plt.plot(recalls, PREGI, 'g', label='EGI (EMD)')
plt.plot(recalls, PRRandom, 'm', label='Random (Euclidean)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Comparison')
plt.legend()
plt.savefig(sys.argv[1])