Shape Descriptors
-----------------

Please view this README rendered by GitHub at https://github.com/bmershon/shape-google

*All images, words, and code contained in this repository may be reproduced so long as the original author is given credit (Chris Tralie and Brooks Mershon).*

This assignment was completed as part of a course in 3D Digital Geometry (Math 290) taken at Duke University during Spring 2016. The course was taught by [Chris Tralie](http://www.ctralie.com/).

### Background

Estimated time spent: 20 hours

*Models from the 20 classes of shapes (each with 10 variants) and their corresponding Extended Gaussian Images. An extended Gaussian image simply bins the normals sampled from a mesh to directions on a sphere. Color is used to indicate where a lot of normals were binned (red is high frequency, blue is low frequency).*

<img src="build/EGI/biplane0.png" width="202">
<img src="build/EGI/biplane0-EGI.png" width="202">
<img src="build/EGI/chair0.png" width="202">
<img src="build/EGI/chair0-EGI.png" width="202">

The purpose of this assignment is to implement functions which take samples from a 3D mesh and produce a signature for a given shape. These signatures take the form of one-dimensonal histograms which may be compared using various metrics, such as Euclidean distance and [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance). A good descriptor will allow shapes to be classified well regardless of their scale and orientation (rotation) in space.

### Features

The following features were implemented:

- Mean-Center / RMS Normalize Point Clouds (6 Points)
- Shell Histograms (6 Points)
- Shell Histograms + Sorted Sectors (10 Points)
- Shell Histograms + PCA Eigenvalues (8 Points)
- D2 Distance Histogram (10 Points)
- A3 Angle Histogram (10 Points)
- Extended Gaussian Image (10 Points)

### Histogram Comparison

The following distance functions were implmented:

- Euclidean Distance (5 Points)
- 1D Earth Mover's Distance (10 Points)

Here is my implementation of Earth Mover's that makes use of broadcasting:

```python
def compareHistsEMD1D(H):
    K = H.shape[0]
    N = H.shape[1]
    D = np.zeros((N, N))
    CDF = np.zeros((K, N))
    CDF[0, :] = H[0, :]
    for i in range(1, K):
        CDF[i, :] = CDF[i - 1, :] + H[i, :]
    # Use transposes and broadcasting so that third dimension can be used to sum
    # corresponding distances.
    D = np.sum(np.absolute(CDF[:, np.newaxis, :].T - CDF.T[np.newaxis, :, :]), axis=2)
    return D
```

### Performance Evaluation (25 points)

#### Self-similarity Matrices

Before generating various precision recall graphs, I found it helpful to look for the general structure produced by each type of histogram and distance metric.

##### Euclidean

In the self-similarity matrices below, blue is *close* and red is *far*. The classes of objects come in groups of 10, so there are 20 contiguous sets of 10 rows, each corresponding to a shape class (e.g., sword, biplane, sedan, table). We hope to see a 10x10 px blue square for each group along the diagonal, with *hotter* colors everywhere else in the same row. This means that a particular group is close to itself and far from others when the histogram signatures are compared using Euclidean distance.

<img src="build/similarity/D2/D2.png" width="405">
<img src="build/similarity/A3/A3.png" width="404">
<img src="build/similarity/EGI/EGI.png" width="405">
<img src="build/similarity/random/random.png" width="405">

We see that for each plot there is a clear structure in which blue squares appear along the diagonal. However, in the case of the **D2 descriptor**, it becomes clear why we see precision recall drop quickly after the first one or two shapes in a class (of 9 other shapes) are found. Looking across one row, we see the dark blue square on the diagonal, but we also see other dark blue pixels spread throughout the row. Therefore, only the first one or two shapes in a class are easily found to be *close* under a metric like Euclidean histogram distance. After the first couple shapes are receovered, many other shapes from outside the class end up having distances from the shape we are comparing it to that are *less* than the distance of the current shape to another member of the same class that has not yet been *recalled*.


The **Extended Gaussian Image descriptor** is interesting, because we see that for some shapes it is working rather well, and for other shapes it is failing to produce a nice 10x10 dark blue square along the diagonal. Why is this? Extended Gaussian Image histograms rely on first finding the principal axes of an object and performing a projection onto a new coordinate system in order to attempt to factor out rotation.

```py
def getEGIHistogram(Ps, Ns, SPoints):
    S = SPoints.shape[1]
    hist = np.zeros(S)

    # align point cloud with PCA axes
    A = np.dot(Ps, Ps.T)
    [eigs, R] = np.linalg.eig(A)
    rotated = np.dot(R.T, Ns)

    D = np.dot(rotated.T, SPoints) # N x M
    nearest = np.argmax(D, 1) # for each normal, the index of nearest spherical direction
    count = np.bincount(nearest) # number ofnormals associated with each direction
    hist[:count.shape[0]] = count
    return hist
```

In the case of the sword family of shapes, we see that the long blade and handle produce a fairly consistent set of principal axes. But other shapes may be more likely to experience poor PCA axes choices on account of their approximate rotational symmetries. If the choice of PCA axes is bad, we see that garbage in will produce garbage out: items within the same class may appear to have signature histograms that are far apart. This explains why the EGI histogram and Euclidean Distance produced a rather inconsistent structure across different classes in the above self-similarity matrix.

##### Earth Mover's Distance

We can also look at D2 and EGI self-similarity matrices when the Earth Mover's Distance is used:

<img src="build/similarity/EMD/D2/EMD-D2.png" width="405">
<img src="build/similarity/EMD/EGI/EMD-EGI.png" width="404">

The D2 self-similarity matrix created using Earth Mover's Distance is not substantially different than the previous D2 image using Euclidean distance. However, the EGI image is substantially different. Interestingly, the image indicates that switching from Euclidean distance to EMD trades problems: with EMD, shapes in the same class are close to each other and to shapes in other classes, where with Euclidean distance shapes in the same class are in many cases too far from each other.

### Precision Recall

#### Different descriptors
The precision recall graphs help summarize the operation of looking down a row of the self-similarity matrices and picking indices from coldest to warmest values until all 9 other shapes in the row's class have been recalled:

```py
def getPrecisionRecall(D, NPerClass = 10):
    PR = np.zeros(NPerClass - 1)
    Recalls = np.zeros((D.shape[0], NPerClass - 1))
    for i in range(D.shape[0]):
        v = np.zeros(NPerClass - 1)
        base = (i // NPerClass) * NPerClass
        group = range(base, base+10)
        index = np.argsort(D[i, :])
        recalled = 0;
        for k in range(len(index)):
            if index[k] in group and index[k] != i:
                recalled = recalled + 1
                v[recalled - 1] = recalled / float(max(k, 1))
        Recalls[i, :] = v
    PR = np.mean(Recalls, axis=0)
    return PR
```

A comparison of various types of histogram functions suggests that D2 is the best performer. One reason why D2 may perform well for this dataset is that D2 does not depend on rotation, so the other methods which attempt to align a model with PCA axes may be thrown off by shapes with rotational symmetries.

It is important to note that A3 is not too much worse than D2, yet EGI is considerably worse than all of the tested metrics. As was noted when we looked at the self-similarity matrices, EGI may suffer from poor PCA alignments due to rotational symmetries in many of the objects.I did find that EMD actually lowered the precision recall for EGI, but it seemed more appropriate to use this descriptor for EGI.

A summary of the unique features of the descriptors:

- **D2**: does not care about rotation. **It is notable that in my tests, EMD caused D2 to produce worse precision recall.**
- **A3**: does not care about rotation, but performed worse than D2 on the first couple of recalled shapes
- **Shell**: does not care about rotation, performed nearly the same as A3 and the sectored shell descriptor
- **Sectored Shell**: does not care about rotation
- **EGI**: subject to poor PCA axes alignment, which makes shapes within the same class appear far under Euclidean, EMD
- **Random**: this control demonstrates that the other descriptors do in fact aid in classifying most of the shapes in a class

<img src="build/precision-recall/compare/precision-recall.png" width="100%">

#### Changing parameters

- Increasing the number of samples improves precision recall for D2 under Euclidean Distance up to about 10,000 samples.
- Binning has little effect on Shell Histograms. *This was surprising.*
- Earth Mover's Distance is actually worse for Extended Gaussian Image classification.
- The number of spherical directions used for EGI under Euclidean distance comparison had little effect on precision-recall.

<img src="build/precision-recall/D2/precision-recall-D2.png" width="405">
<img src="build/precision-recall/shell/precision-recall-shell.png" width="405">
<img src="build/precision-recall/EMD/precision-recall-EMD.png" width="405">
<img src="build/precision-recall/EGI/precision-recall-EGI.png" width="405">

### Classification Contest

Given more time, I had hoped to tinker with weighted combinations of various histograms by looking at self-similarity matrices resulting from their weighted sums. I also had the idea to use **clipped/filtered functions** of various distance matrices as well as **higher-order matrices** resulting from applying global operations such as taking the power of each element in a distance matrix.

I did make a submission in the form of implementing `getMyShapeDistances`:

```py
def getMyShapeDistances(PointClouds, Normals):
    HistsD2 = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 10000)
    D0 = compareHistsEuclidean(HistsD2)

    HistsA3 = makeAllHistograms(PointClouds, Normals, getA3Histogram, 30, 10000)
    D1 = compareHistsEuclidean(HistsA3)

    HistsEGI = makeAllHistograms(PointClouds, Normals, getEGIHistogram, getSphereSamples(1))
    D2 = compareHistsEMD1D(HistsD2)

    z = float(np.sum(D0) + np.sum(D1) + np.sum(D2))
    D = (D0*np.sum(D0) / z) + (D1*np.sum(D1) / z) + (D2 * np.sum(D2) / z)
    return D
```

*This weighting scheme doesn't exactly do much better than D2 and Euclidean distance.* After reading the [Funkhouser paper](http://dpd.cs.princeton.edu/Papers/FunkShapeTog.pdf) on spherical harmonics (which, unlike EGI, don't suffer from poor PCA axes alignment), I believe it would be necessary to at least incorporate D2 and spherical harmonic histograms into a weighting scheme. I am still unsure as to which types of histogram comparators are best for this classification task. My empirical results, which may well be affected by errors in my implementation, show that Earth Mover's distance was worse than Euclidean distance for D2 and A3.


Here's the precision recall for a weighted combination of D2, A3, and EGH (EMD). Better than the control, but not spectacular, either.

<img src="build/contest/mershon-contest.png" width="405">

### Note

I used a Makefile to automate the process of creating output graphs and models. Makefiles are awesome. When I type make, every file that I want for my report is built if it doesn't already exist. This made the testing process less painful.

There is still too much code duplication in the test files, and the `ShapeStatistics.py` file could be broken up into modules with more appropriate namespacing. Given more time to work on this project, a reorganization of the shape statics functionality would have been my next task for myself.

I am keen to talk to Roger (partnered with Joy Patel) during the next unit to see how he implemented spherical harmonics (in NumPy). Chris Tralie mentioned that this was something he would have to work on for his final project. I dind't get to that task this time around.

I believe my implementation hits between 100 and 105 (contest submission) out of the 100 points needed for a group with two or fewer people.
