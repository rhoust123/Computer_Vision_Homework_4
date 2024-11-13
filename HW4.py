# Import libraries
import os
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import img_as_float32
from skimage import io
import matplotlib.pyplot as plt
from keras.datasets import mnist
import statistics
from tqdm import tqdm

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



def Problem_Two():

    for i in range(0,3):
        # Read in img
        img = cv2.imread('images/csm{}.jpg'.format(i+1))
        assert (img is not None), 'cannot read given image'

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # trained data: https://github.com/opencv/opencv/tree/master/data/haarcascades
        haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = haar_face.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3) 
        
        for (x, y, w, h) in faces: 
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) 

        cv2.imshow('Faces', img) 
        cv2.waitKey(0) 

    return 

def Problem_Three():

    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(X_train[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_train[i]))
    plt.show()


    # Data preprocessing
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)


    # KNN classifier
    def distance(x,y):
        # define some distance function

        # L2 Norm (Euclidean)
        return np.linalg.norm(np.array(x) - np.array(y))

    def kNN(x, k, data, label):
        #list of distances between the given image and the images of the training set
        distances = [distance(x,data[i]) for i in range(len(data))]

        distances = np.array(distances)

        # find the k nearest neighbors
        lowest_k = np.argsort(distances)[:k]

        predictions = []

        # get predictions based on those neighbors
        for i in range(len(lowest_k)): 
            predictions.append(label[lowest_k[i]])

        clas = statistics.mode(predictions)

        return clas # estimated class

    def image_show(i, data, label, clas):
        x = data[i] # get vectorized image
        x = x.reshape((28,28)) # reshape it into 28x28 format
        title = 'predicted={0:d}, true={0:d}'.format(clas, label[i])
        plt.imshow(x, cmap='gray') 
        plt.title(title)
        plt.show()


    # Single Test case
    i = 10
    clas = kNN(X_test[i], 5, X_train, y_train)
    image_show(i, X_test, y_test, clas)

    true_pos = 0
    false_pos = 0

    # Precision = True Positive / (True Positive + False Positive)
    for i in tqdm(range(1000), desc="Testing Samples"):

        clas = kNN(X_test[i], 5, X_train, y_train)

        # True Positives
        if clas == y_test[i]: 
            true_pos += 1
        
        # False Positives
        else:
            false_pos += 1

    # Precision on test data
    print('precision = ', (true_pos) / (true_pos + false_pos)) #precision)




# def Problem_Four():
# def Problem_Five():

def main():
    #Problem_One()
    #Problem_Two()
    Problem_Three()
    # Problem_Four()
    # Problem_Five()

    return 0

main()