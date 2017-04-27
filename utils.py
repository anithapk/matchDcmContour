import dicom
import numpy as np
from PIL import Image, ImageDraw
import glob, os, shutil, sys
import csv as csv
from sklearn.utils import shuffle
from parsing import *
import matplotlib.pyplot as plt

def overlayImg(dcmImg,mask,alpha=0.7):
    # ---input: grayscale dicom dict, boolean mask, alpha for fusing the images
    # ---output: uint8  RGB fusedImg  
    # going from int16 to uint8 through float64
    dImg = dcmImg['pixel_data']/np.max(dcmImg['pixel_data'])
    dImg_3C = np.dstack((dImg,dImg,dImg))*255
    dImg_3C = dImg_3C.astype("uint8")
    maskR = mask.astype('uint8')
    maskR[np.where(mask==1)]=255
    cmask = np.dstack((maskR,mask.astype("uint8"),mask.astype("uint8")))
    # overlay the contour in red color
    combImg = dImg_3C
    combImg[:,:,0][np.where(mask==1)] = alpha*combImg[:,:,0][np.where(mask==1)] + (1-alpha)*cmask[:,:,0][np.where(mask==1)] 
    return combImg

def createContMask(contFname,imSize):
    #---input: filename of the contour file including the full path,
    #---imSize: [0]: num rows, [1]: num cols
    #---output: bool mask
    xyCoord = parse_contour_file(contFname)
    contMask = poly_to_mask(xyCoord, imSize[1], imSize[0])
    #contMask = np.zeros(img['pixel_data'].shape).astype('uint8')
    #contMask[np.floor(ycoord).astype('int'),np.floor(xcoord).astype('int')] = 1
    #dilate and fill to get mask
    return contMask

def getNumFromDcmFname(dcmFname):
    #---input: dicom filename including the full path
    #---output: get the image number from the filename
    return int(os.path.split(dcmFname)[1].split('.')[0])

def getNumFromContFname(contFname):
    #---input: filename of the contour file including the full path
    #---output: get the image number from the filename
    return int(os.path.split(contFname)[1].split('-')[2])

def writeCSV(listVal,fName):
    #---input: listVal - variable to be written as csv file, fName of the csv file
    try:
        f = open(fName,'w')
    except IOError:
        print('cannot open file for writing dcm,cont fname pair')
        sys.exit(1)
    else:
        with f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(listVal)):
               writer.writerow(listVal[i])
            f.close()
    return

def genTrainPair(dcmDir,conDir,opFname,getNumFromDcmFname=getNumFromDcmFname,getNumFromContFname=getNumFromContFname):
    #---input: dicom filename including the full path,
    #---conDir: filename of the contour file including the full path,
    #---output: csv filename for writing dicom and contour pairs including the full path
    conFnames = glob.glob(conDir+'/*.txt')
    imgFnames = glob.glob(dcmDir+'/*.dcm')
    imgNum = [getNumFromDcmFname(i) for i in imgFnames]
    imgNum = np.array(imgNum)
    imgPath = {imgNum[i]:imgFnames[i] for i in range(len(imgNum))}
    trainpair = []
    for n,i in enumerate(conFnames):
       curContNum = getNumFromContFname(i)
       # pull the corresponding image
       trainpair.append([imgPath[curContNum],i])
    writeCSV(trainpair,opFname)
    return

def chkMask(dcmDir,conDir,opBaseDir):
    #---input: dicom filename including the full path,
    #---conDir: filename of the contour file including the full path,
    #---opBaseDir: base directory where the fused images are saved
    try:
        pairCSV = np.genfromtxt(os.path.join(opBaseDir,'imgContPair.csv'), delimiter=',', 
                                usecols=(0,1), dtype=str,skip_header=True)
    except IOError:
        print("Unable to find/open imgContPair.csv")
        sys.exit(1)
    # set the dir for writing the overlay files
    opDir = os.path.join(opBaseDir,'overlay')
    if not os.path.exists(opDir):
        try:
            os.makedirs(opDir)
        except shutil.Error:
            print("cannot create directory for saving overlay images")           
    else:
        try:
            shutil.rmtree(opDir)   
        except shutil.Error:
            print("cannot delete overlay directory")
            sys.exit(1)
        try:
            os.makedirs(opDir)
        except shutil.Error:
            print("cannot create directory for saving overlay images")
            sys.exit(1)
    plt.figure(figsize=(10,16))
    n=0
    for i in range(len(pairCSV)):
        curImg = parse_dicom_file(pairCSV[i][0])
        contMask = createContMask(pairCSV[i][1],curImg['pixel_data'].shape)
        overImg = overlayImg(curImg,contMask)
        overFname = os.path.join(opDir,str(getNumFromDcmFname(pairCSV[i][0]))+'.jpg')
        Image.fromarray(overImg).save(overFname)
        plt.subplot(5,5,n+1)
        n = n+1
        plt.imshow(overImg); plt.axis('off')

class readTrainPair:
    # class for generating train,mask pairs
    # input: dicom filename including the full path,
    # contFnames: filename of the contour file including the full path,
    # batch_size, concatDim: concatenate images along 0(x) or 2(z)
    # ouput: iterator that gives batch_size dicom img, boolean mask pairs
    def __init__(self, imgFnames, contFnames,batch_size,concatDim):
        self.imgFnames = imgFnames
        self.contFnames = contFnames
        self.batchSize = batch_size
        self.concatDim = concatDim
        self.current = 0
        self.high = len(imgFnames)
    
    def __iter__(self):
        return self
    
    def __next__(self): 
        if self.current >= self.high:
            raise StopIteration
        else:
            print("lastIndex:",self.current)
            img0 = parse_dicom_file(self.imgFnames[self.current])
            cont0 = createContMask(self.contFnames[self.current],img0['pixel_data'].shape)
            if (self.current+self.batchSize)>self.high:
                end = len(self.imgFnames)
            else:
                end = self.current+self.batchSize
            
            if self.concatDim==0:
                imgArray = np.zeros([self.batchSize,img0['pixel_data'].shape[0],
                                     img0['pixel_data'].shape[1]])
                imgArray[0] = img0['pixel_data']
                contArray = np.zeros(imgArray.shape).astype(bool)
                contArray[0] = cont0
                n = 1
                for i in range(self.current+1,end,1):
                    tmpImg = parse_dicom_file(self.imgFnames[i])
                    imgArray[n] = tmpImg['pixel_data']
                    contArray[n] = createContMask(self.contFnames[i],tmpImg['pixel_data'].shape)
                    n=n+1
            elif concatDim==2:
                imgArray = np.zeros([img0['pixel_data'].shape[0],img0['pixel_data'].shape[1],self.batchSize])
                imgArray[:,:,0] = img0['pixel_data']
                contArray = np.zeros(imgArray.shape).astype(bool)
                contArray[:,:,0] = cont0
                n = 1
                for i in range(self.lastIndex+1,end,1):
                    tmpImg = parse_dicom_file(self.imgFnames[i])
                    imgArray[:,:,i] = tmpImg['pixel_data']
                    contArray[:,:,i] = createContMask(self.contFnames[i],tmpImg['pixel_Data'].shape)
                    n=n+1
            else:
                print("unknown dimension for concatenating images")
            self.current += self.batchSize
        return imgArray,contArray
