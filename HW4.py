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
    
    for i in range(0,3):
        # Read in img
        img = cv2.imread('images/csm{}.jpg'.format(i+1))
        assert (img is not None), 'cannot read given image'

        # create HOG descriptor 
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # img = image to detect objects in
        # window = cell size
        # padding = padding in detection window
        # scale = image scaling for img pyramid
        (regions, confidence) = hog.detectMultiScale(img, winStride=(3,3), padding=(2,2), scale=1.75)
        
        # Drawing the regions in the Image
        for (x, y, w, h) in regions:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Pedestrians", img)
        cv2.waitKey(0)

    # Issues w/ Pedestrian Detection: 
    # 1. The first issue I noticed was that, at times, people's legs are detected as "pedestrians." I tried many different combinations of scale, padding, and winStride, and it seemed like I would get a "leg" detection most consistently - even at times where there were no pedestrian detections. 
    # 2. Another issue I noticed was that pedestrians that were slightly obscured were not detected very reliably. For example, in the image csm3.jpg, there is a person walking slightly behind another that has about half his body obstructed. I could not get the HOG detector to recognize this person as a pedestrian. 
    # 3. (Bonus!) I also noticed that in the first image, csm1.jpg, there was an alarming absence of pedestrian detection altogether, where in csm2 and csm3 there were many. I think this may be because the people in the image were dynamically posed and were standing together - to the HOG descriptor, they may have looked like a shapeless amoeba as opposed to what a pedestrian should look like. 



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