import cv2

image = cv2.imread('Inputs/Bender.png',0)

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
padding = (8,8)
locations = ((10,20),)
hist = hog.compute(image,winStride,padding,locations)

print(hist)