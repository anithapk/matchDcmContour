import numpy as np
from PIL import Image, ImageDraw
import glob, os, shutil,sys
import csv as csv
from sklearn.utils import shuffle
from parsing import *
from utils import *
import matplotlib.pyplot as plt

def pairImgCont(baseDir):
    try:
        ipDir = np.genfromtxt(os.path.join(baseDir,'link.csv'), delimiter=',', usecols=(0,1), dtype=str,skip_header=True)
    except IOError:
        print("unable to read/find the link csv file")
        sys.exit(1)
    for ind in range(len(ipDir)):
        curDcmDir = os.path.join(baseDir,'dicoms/'+ipDir[ind][0])
        curConDir = os.path.join(baseDir,'contourfiles/'+ipDir[ind][1]+"/i-contours")
        trainCSV = os.path.join(os.path.split(curConDir)[0],'imgContPair.csv')
        bDir = os.path.split(curConDir)[0]
        laneMask = genTrainPair(curDcmDir,curConDir,trainCSV)
        #chkMask(curDcmDir,curConDir,bDir)

def main():
    if len(sys.argv[1:])==0:
        print("provide the base folder containing images and contour files")
        sys.exit(1)
    else:
        baseDir = sys.argv[1]
    pairImgCont(baseDir)

if __name__ == "__main__":
    main()
