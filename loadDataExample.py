# This is the code for PS4 of Dr. Parikh's computer vision class.
# the code use part of the the polygonal region selection tools from https://github.com/jdoepfert/roipoly.py

import glob
# This script is to illustrate a couple of the provided functions, and to demonstrated loading a data file.
import os
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.color import rgb2gray

from displaySIFTPatches import displaySIFTPatches
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from selectRegion import roipoly

# specific frame dir and siftdir
framesdir = "frames/"
siftdir = "sift/"

fnames = glob.glob(siftdir + "*.mat")
fnames = [os.path.basename(name) for name in fnames]
print(f"reading {len(fnames)} files...", flush=True)

# select a random file
fname = random.choice(fnames)
print("reading {fname}", flush=True)

# load that file
mat = scipy.io.loadmat(os.path.join(siftdir, fname))
num_feats = mat["descriptors"].shape[0]

print(f"{fname} contains {num_feats} features", flush=True)
print(f"feature dimension:  {mat['descriptors'].shape[1]}", flush=True)

# read the associated image
im = imageio.imread(os.path.join(framesdir, fname[:-4]))

# display the image and its SIFT features drawn as squares
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(im)

coners = displaySIFTPatches(mat["positions"], mat["scales"], mat["orients"])
for j in range(len(coners)):
    ax.plot(
        [coners[j][0][1], coners[j][1][1]],
        [coners[j][0][0], coners[j][1][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        [coners[j][1][1], coners[j][2][1]],
        [coners[j][1][0], coners[j][2][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        [coners[j][2][1], coners[j][3][1]],
        [coners[j][2][0], coners[j][3][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        [coners[j][3][1], coners[j][0][1]],
        [coners[j][3][0], coners[j][0][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
ax.set_xlim(0, im.shape[1])
ax.set_ylim(0, im.shape[0])
plt.gca().invert_yaxis()
plt.show()

# now display the same image but only show N of the features

N = 10  # to visualize a sparser set of the features
rand_indices = np.random.permutation(num_feats)[:N]
print(f"displaying {N} random features...", flush=True)

fig = plt.figure()
bx = fig.add_subplot(111)
bx.imshow(im)
coners = displaySIFTPatches(
    mat["positions"][rand_indices, :],
    mat["scales"][rand_indices, :],
    mat["orients"][rand_indices, :],
)
for j in range(len(coners)):
    bx.plot(
        [coners[j][0][1], coners[j][1][1]],
        [coners[j][0][0], coners[j][1][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    bx.plot(
        [coners[j][1][1], coners[j][2][1]],
        [coners[j][1][0], coners[j][2][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    bx.plot(
        [coners[j][2][1], coners[j][3][1]],
        [coners[j][2][0], coners[j][3][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    bx.plot(
        [coners[j][3][1], coners[j][0][1]],
        [coners[j][3][0], coners[j][0][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
bx.set_xlim(0, im.shape[1])
bx.set_ylim(0, im.shape[0])
plt.gca().invert_yaxis()
plt.show()

# now show how to select a subset of the features using polygon drawing.
print(
    "now use the mouse to draw a polygon, right click or double click to end it",
    flush=True,
)
plt.imshow(im)
roi = roipoly(color="r")
indices = roi.get_indices(im, mat["positions"])

np.save("points.npy", indices)

# indices contains the indices of the SIFT features whose centers fall
# within the selected region of interest.
# Note that these indices apply to the *rows* of 'descriptors' and
# 'positions', as well as the entries of 'scales' and 'orients'
# now display the same image but only in the polygon.
print("displying features in the selected region", flush=True)
fig = plt.figure()
bx = fig.add_subplot(111)
bx.imshow(im)

coners = displaySIFTPatches(
    mat["positions"][indices, :], mat["scales"][indices, :], mat["orients"][indices, :]
)
for j in range(len(coners)):
    bx.plot(
        [coners[j][0][1], coners[j][1][1]],
        [coners[j][0][0], coners[j][1][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    bx.plot(
        [coners[j][1][1], coners[j][2][1]],
        [coners[j][1][0], coners[j][2][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    bx.plot(
        [coners[j][2][1], coners[j][3][1]],
        [coners[j][2][0], coners[j][3][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
    bx.plot(
        [coners[j][3][1], coners[j][0][1]],
        [coners[j][3][0], coners[j][0][0]],
        color="g",
        linestyle="-",
        linewidth=1,
    )
bx.set_xlim(0, im.shape[1])
bx.set_ylim(0, im.shape[0])
plt.gca().invert_yaxis()
plt.show()

# extract an image patch
print("displaying an image patch for one of the first 10 features", flush=True)

patch_num = random.choice(range(min(num_feats, 10)))
img_patch = getPatchFromSIFTParameters(
    mat["positions"][patch_num, :],
    mat["scales"][patch_num],
    mat["orients"][patch_num],
    rgb2gray(im),
)

plt.imshow(img_patch, cmap="gray")
plt.show()
