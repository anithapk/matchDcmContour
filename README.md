# matchDcmContour
The objective of this project is to prepare the dicom images and matching contours for training a convolutional neural network.

Added matchImgCont.py:

\1) overlayImg(dcmImg,mask,alpha=0.7): fuses the dicom grayscale and boolean masks

\2) createContMask(contFname,imSize): creates a boolean mask from the contour file and image size

\3) getNumFromDcmFname(dcmFname) & getNumFromContFname(contFname): these are functions to extract the image number from the filenames. I separated this as a function as it'll be easier to modify in the format of filename changes in the future

\4) genTrainPair(dcmDir,conDir,opFname): main function that takes dicom image and contour directory and creates a list with the pairs.

\5) writeCSV(listVal,fName): write the csv list to file. The files are written individually per patient, which might be easily scalable and one can read a sub-set if needed.

\6) chkMask(dcmDir,conDir,opBaseDir): a debugging function to read the csv pair file, create fused images and write to an overlay folder per patient

\7) class readTrainPair: iterator that takes the dcmFnames and contFnames and returns the images in batches

matchDcmCont.ipynb: notebook for exploration and visualization of the fused images

matchImgCont.py: python script that loops through pairs in link.csv given a base directory. Assumes a directory structure
