import numpy as np
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from selectRegion import roipoly
import glob
# This script is to illustrate a couple of the provided functions, and to demonstrated loading a data file.
import os
import scipy.io
import imageio
import random

#mat = sio.loadmat("twoFrameData.mat")
#im = mat['im1']
#position1 = mat['positions1']
framesdir = "frames/"
siftdir = "sift/"

fnames = glob.glob(siftdir + "*.mat")
fnames = [os.path.basename(name) for name in fnames]

mat = scipy.io.loadmat(os.path.join(siftdir, fnames[200]))
im = imageio.imread(os.path.join(framesdir, fnames[200][:-4]))
position1 = mat['positions']
plt.imshow(im)
roi = roipoly(color="r")

indices = roi.get_indices(im, position1)
roi.display_roi()
pTrans = np.zeros((2, len(roi.all_x_points)))
pTrans[0] = np.array(roi.all_x_points)
pTrans[1] = np.array(roi.all_y_points)
points = np.transpose(pTrans)

np.save("points2.npy", indices)
np.save("region2.npy", points)
