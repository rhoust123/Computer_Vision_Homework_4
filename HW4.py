# Import libraries
import os
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import img_as_float32
from skimage import io

def Problem_One():
  
    img = cv2.imread('images/csm3.jpg')
    assert (img is not None), 'cannot read given image'

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    (regions, _) = hog.detectMultiScale(img, ...)
    
    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(img,...)
    
    cv2.imshow("Pedestrians", img)
    cv2.waitKey(0)


    return 

# def Problem_Two():
# def Problem_Three():
# def Problem_Four():
# def Problem_Five():

def main():
    Problem_One()
    # Problem_Two()
    # Problem_Three()
    # Problem_Four()
    # Problem_Five()

    return 0

main()