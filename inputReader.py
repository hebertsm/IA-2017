import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('Inputs/hsf_0_00001.png',0)

image3 = cv2.imread('Inputs/hsf_1_01070.png',0)

image2 = cv2.imread('Inputs/hsf_0_00000.png',0)


#Detection window size. Align to block size and block stride.
winSize = (64,64)
#Block size in pixels. Align to cell size.
blockSize = (16,16)
#Block stride. It must be a multiple of cell size.
blockStride = (8,8)
#Cell size. Only (8, 8) is supported for now.
cellSize = (8,8)
#Number of bins.
nbins = 9
derivAperture = 1
#Gaussian smoothing window parameter.
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
#Flag to specify whether the gamma correction preprocessing is required or not.
gammaCorrection = 0
#Maximum number of detection window increases.
nlevels = 64

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)


winStride = (8,8)
padding = (0,0)
locations = ((0,0),)

hist1 = hog.compute(image,winStride,padding,locations)
hist2 = hog.compute(image2,winStride,padding,locations)
hist3 = hog.compute(image3,winStride,padding,locations)
#hist = hog.compute(image, winStride)

"""
plt.hist(hist2)
plt.hist(hist1)
plt.show()
"""

print(hist1)

for x in hist1:
    print(x)

#a = np.concatenate((hist1, hist2), axis=0)
#print(np.concatenate((hist1, hist2), axis=0))
#m = np.asmatrix(hist1)



