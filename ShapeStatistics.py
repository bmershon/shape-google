#Purpose: To implement a suite of 3D shape statistics and to use them for point
#cloud classification
#TODO: Fill in all of this code for group assignment 2
import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *

import numpy as np
import matplotlib.pyplot as plt

POINTCLOUD_CLASSES = ['biplane', 'desk_chair', 'dining_chair', 'fighter_jet', 'fish', 'flying_bird', 'guitar', 'handgun', 'head', 'helicopter', 'human', 'human_arms_out', 'potted_plant', 'race_car', 'sedan', 'shelves', 'ship', 'sword', 'table', 'vase']

NUM_PER_CLASS = 10

#########################################################
##                UTILITY FUNCTIONS                    ##
#########################################################

#Purpose: Export a sampled point cloud into the JS interactive point cloud viewer
#Inputs: Ps (3 x N array of points), Ns (3 x N array of estimated normals),
#filename: Output filename
def exportPointCloud(Ps, Ns, filename):
    N = Ps.shape[1]
    fout = open(filename, "w")
    fmtstr = "%g" + " %g"*5 + "\n"
    for i in range(N):
        fields = np.zeros(6)
        fields[0:3] = Ps[:, i]
        fields[3:] = Ns[:, i]
        fout.write(fmtstr%tuple(fields.flatten().tolist()))
    fout.close()

#Purpose: To sample a point cloud, center it on its centroid, and
#then scale all of the points so that the RMS distance to the origin is 1
def samplePointCloud(mesh, N):
    (Ps, Ns) = mesh.randomlySamplePoints(N)
    centroid = getCentroid(Ps)
    Ps = Ps - centroid
    scale = 1 / (np.sqrt(np.sum(np.square(Ps) / N)))
    Ps = np.multiply(scale, Ps)
    RMS = np.sqrt(np.sum(np.square(Ps)) / N)
    print("mesh translated by \n %s\nscaled by %s, RMS is %s" % (centroid, scale, RMS))
    return (Ps, Ns)

# Returns a 3 x 1 matrix
def getCentroid(PC):
    # mean of column vectors (axis 1) 
    return np.mean(PC, 1, keepdims=True)

def length(u):
    return np.sqrt(np.sum(np.square(u)))

def dot(u, v):
    return np.dot(u, v)

def angle(a, b, c):
    u = b - a
    v = c - a
    return np.arccos(dot(u, v) / (length(u) * length(v)))

#Purpose: To sample the unit sphere as evenly as possible.  The higher
#res is, the more samples are taken on the sphere (in an exponential 
#relationship with res).  By default, samples 66 points
def getSphereSamples(res = 2):
    m = getSphereMesh(1, res)
    return m.VPos.T

#Purpose: To compute PCA on a point cloud
#Inputs: X (3 x N array representing a point cloud)
def doPCA(X):
    D = X.dot(X.T) # X*X Transpose
    (eigs, V) = np.linalg.eig(D) # Eigenvectors in columns
    return (eigs, V)

# Fisher-Yates linear in-place shuffle
def shuffle(array):
    m = len(array) - 1
    # while there are elements to shuffle
    while m > 0:
        # pick a random element from the end
        i = np.random.randint(m, size=1)[0]
        t = array[i]
        array[i] = array[m]
        array[m] = t
        m = m - 1

#########################################################
##                SHAPE DESCRIPTORS                    ##
#########################################################

#Purpose: To compute a shape histogram, counting points
#distributed in concentric spherical shells centered at the origin
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency)
#NShells (number of shells), RMax (maximum radius)
#Returns: hist (histogram of length NShells)
def getShapeHistogram(Ps, Ns, NShells, RMax):
    hist = np.zeros(NShells)
    bins = np.square(np.linspace(0.0, RMax, NShells))
    indices = np.digitize(np.sum(np.multiply(Ps, Ps), axis=0), bins) - 1
    count = np.bincount(indices)
    hist[:count.shape[0]] = count
    
    return hist
    
#Purpose: To create shape histogram with concentric spherical shells and
#sectors within each shell, sorted in decreasing order of number of points
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NShells (number of shells), 
#RMax (maximum radius), SPoints: A 3 x S array of points sampled evenly on 
#the unit sphere (get these with the function "getSphereSamples")
def getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints):
    NSectors = SPoints.shape[1] # number of spherical samples
    hist = np.zeros((NShells, NSectors))    
    bins = np.square(np.linspace(0.0, RMax, NShells))
    indices = np.digitize(np.sum(np.multiply(Ps, Ps), axis=0), bins) - 1
    for i in range(NShells):
        subset = Ps[:, indices == i]
        D = np.dot(subset.T, SPoints) # N x M
        nearest = np.argmax(D, 1) # for each point, the index of nearest spherical direction
        count = np.bincount(nearest) # points associated with each direction
        hist[i, :count.shape[0]] = np.sort(count)[::-1] 
    
    return hist.flatten() #Flatten the 2D histogram to a 1D array

#Purpose: To create shape histogram with concentric spherical shells and to 
#compute the PCA eigenvalues in each shell
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NShells (number of shells), 
#RMax (maximum radius), sphereRes: An integer specifying points on thes phere
#to be used to cluster shells
def getShapeHistogramPCA(Ps, Ns, NShells, RMax):
    hist = np.zeros((NShells, 3))
    bins = np.square(np.linspace(0.0, RMax, NShells)) # squared radii
    indices = np.digitize(np.sum(np.square(Ps), axis=0), bins) - 1
    for i in range(NShells):
        subset = Ps[:, indices == i]
        D = np.dot(subset, subset.T)
        (eigs, V) = np.linalg.eig(D) # Eigenvectors in columns
        hist[i, :eigs.shape[0]] = np.sort(eigs)[::-1] # decreasing eigenvalues
    return hist.flatten() #Flatten the 2D histogram to a 1D array

#Purpose: To create shape histogram of the pairwise Euclidean distances between
#randomly sampled points in the point cloud
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), DMax (Maximum distance to consider), 
#NBins (number of histogram bins), NSamples (number of pairs of points sample
#to compute distances)
def getD2Histogram(Ps, Ns, DMax, NBins, NSamples):
    N = NSamples * 2
    if N > Ps.shape[1]: N = (Ps.shape[1] // 2) * 2
    hist = np.zeros(NBins)
    bins = np.square(np.linspace(0.0, DMax, NBins)) # squared distances
    perm = np.arange(Ps.shape[1])
    shuffle(perm) # permutation of indices to sample
    sample = Ps[:, perm[:N]] # sample only 2 x N points
    a = sample[:, 0::2] # evens
    b = sample[:, 1::2] # odds
    distances = np.sum(np.square(a - b), axis=0) # squared distances
    indices = np.digitize(distances, bins) - 1
    count = np.bincount(indices)
    hist[:count.shape[0]] = count
    return hist

#Purpose: To create shape histogram of the angles between randomly sampled
#triples of points
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NBins (number of histogram bins), 
#NSamples (number of triples of points sample to compute angles)
def getA3Histogram(Ps, Ns, NBins, NSamples):
    N = NSamples * 3
    if N > Ps.shape[1]: N = (Ps.shape[1] // 3) * 3
    hist = np.zeros(NBins)
    bins = np.square(np.linspace(0.0, 2 * np.pi, NBins)) # squared distances
    perm = np.arange(Ps.shape[1])
    shuffle(perm) # permutation of indices to sample
    sample = Ps[:, perm[:N]] # sample only 2 x N points
    # a = sample[:, 0::3] 
    # b = sample[:, 1::3]
    # c = sample[:, 2::3]
    # angles = 
    return hist

#Purpose: To create the Extended Gaussian Image by binning normals to
#sphere directions after rotating the point cloud to align with its principal axes
#Inputs: Ps (3 x N point cloud) (use to compute PCA), Ns (3 x N array of normals), 
#SPoints: A 3 x S array of points sampled evenly on the unit sphere used to 
#bin the normals
def getEGIHistogram(Ps, Ns, SPoints):
    S = SPoints.shape[1]
    hist = np.zeros(S)
    ##TOOD: Finish this; fill in hist
    return hist

#Purpose: To create an image which stores the amalgamation of rotating
#a bunch of planes around the largest principal axis of a point cloud and 
#projecting the points on the minor axes onto the image.
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals, not needed here),
#NAngles: The number of angles between 0 and 2*pi through which to rotate
#the plane, Extent: The extent of each axis, Dim: The number of pixels along
#each minor axis
def getSpinImage(Ps, Ns, NAngles, Extent, Dim):
    #Create an image
    hist = np.zeros((Dim, Dim))
    #TODO: Finish this
    return hist.flatten()


#Purpose: To create a histogram of spherical harmonic magnitudes in concentric
#spheres after rasterizing the point cloud to a voxel grid
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals, not used here), 
#VoxelRes: The number of voxels along each axis (for instance, if 30, then rasterize
#to 30x30x30 voxels), Extent: The number of units along each axis (if 2, then 
#rasterize in the box [-1, 1] x [-1, 1] x [-1, 1]), NHarmonics: The number of spherical
#harmonics, NSpheres, the number of concentric spheres to take
def getSphericalHarmonicMagnitudes(Ps, Ns, VoxelRes, Extent, NHarmonics, NSpheres):
    hist = np.zeros((NSpheres, NHarmonics))
    #TODO: Finish this
    
    return hist.flatten()

#Purpose: Utility function for wrapping around the statistics functions.
#Inputs: PointClouds (a python list of N point clouds), Normals (a python
#list of the N corresponding normals), histFunction (a function
#handle for one of the above functions), *args (addditional arguments
#that the descriptor function needs)
#Returns: AllHists (A KxN matrix of all descriptors, where K is the length
#of each descriptor)
def makeAllHistograms(PointClouds, Normals, histFunction, *args):
    N = len(PointClouds)
    #Call on first mesh to figure out the dimensions of the histogram
    h0 = histFunction(PointClouds[0], Normals[0], *args)
    K = h0.size
    AllHists = np.zeros((K, N))
    AllHists[:, 0] = h0
    for i in range(1, N):
        print "Computing histogram %i of %i..."%(i+1, N)
        AllHists[:, i] = histFunction(PointClouds[i], Normals[i], *args)
    return AllHists

#########################################################
##              HISTOGRAM COMPARISONS                  ##
#########################################################

#Purpose: To compute the euclidean distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the Euclidean
#distance between the histogram for point cloud i and point cloud j)
def compareHistsEuclidean(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    return D

#Purpose: To compute the cosine distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the cosine
#distance between the histogram for point cloud i and point cloud j)
def compareHistsCosine(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    return D

#Purpose: To compute the cosine distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the chi squared
#distance between the histogram for point cloud i and point cloud j)
def compareHistsChiSquared(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    return D

#Purpose: To compute the 1D Earth mover's distance between a set
#of histograms (note that this only makes sense for 1D histograms)
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the earth mover's
#distance between the histogram for point cloud i and point cloud j)
def compareHistsEMD1D(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    return D


#########################################################
##              CLASSIFICATION CONTEST                 ##
#########################################################

#Purpose: To implement your own custom distance matrix between all point
#clouds for the point cloud clasification contest
#Inputs: PointClouds, an array of point cloud matrices, Normals: an array
#of normal matrices
#Returns: D: A N x N matrix of distances between point clouds based
#on your metric, where Dij is the distnace between point cloud i and point cloud j
def getMyShapeDistances(PointClouds, Normals):
    #TODO: Finish this
    #This is just an example, but you should experiment to find which features
    #work the best, and possibly come up with a weighted combination of 
    #different features
    HistsD2 = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 100000)
    DEuc = compareHistsEuclidean(HistsD2)
    return DEuc

#########################################################
##                     EVALUATION                      ##
#########################################################

#Purpose: To return an average precision recall graph for a collection of
#shapes given the similarity scores of all pairs of histograms.
#Inputs: D (An N x N matrix, where the ij entry is the earth mover's distance
#between the histogram for point cloud i and point cloud j).  It is assumed
#that the point clouds are presented in contiguous chunks of classes, and that
#there are "NPerClass" point clouds per each class (for the dataset provided
#there are 10 per class so that's the default argument).  So the program should
#return a precision recall graph that has 9 elements
#Returns PR, an (NPerClass-1) length array of average precision values for all 
#recalls
def getPrecisionRecall(D, NPerClass = 10):
    PR = np.zeros(NPerClass-1)
    #TODO: Finish this, compute average precision recall graph
    #using all point clouds as queries
    return PR

#########################################################
##                     MAIN TESTS                      ##
#########################################################

if __name__ == '__main__':  
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
            (Ps, Ns) = samplePointCloud(m, NRandSamples)
            PointClouds.append(Ps)
            Normals.append(Ps)
    
    #TODO: Finish this, run experiments.  Also in the above code, you might
    #just want to load one point cloud and test your histograms on that first
    #so you don't have to wait for all point clouds to load when making
    #minor tweaks
