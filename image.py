import sys, os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def mergeTwoDirs(imgdir1, imgdir2, outdir):
    namelist = os.listdir(imgdir1)
    for imname in namelist:
        imgpath1 = os.path.join(imgdir1, imname)
        imgpath2 = os.path.join(imgdir2, imname)
        img1 = cv2.imread(imgpath1)
        img2 = cv2.imread(imgpath2)
        both = np.hstack((img1,img2))

        outpath = os.path.join(outdir, imname)
        cv2.imwrite(outpath, both)
        #vis0 = cv2.fromarray(both)
        #cv2.imshow("show", both)
        #cv2.waitKey()

def drawHeatmapAndImage(img, htmap):
    #show image
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    plt.imshow(img)
    #show heatmap
    nrow = htmap.shape[0]
    ncol = htmap.shape[1]
    X,Y = np.meshgrid([x for x in xrange(ncol)], [x for x in xrange(nrow)])
    a = fig.add_subplot(1,2,2)
    plt.imshow(htmap)
    plt.grid(True)
    plt.colorbar()
    plt.show()


