from __future__ import print_function

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

import argparse
import time
import imutils
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(network, num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
    return network

def predict_label(image,model='model.npz'):
    input_var = T.tensor4('image')
    network = build_cnn(input_var)
    with np.load(model) as f:
        param_values = [f['arr_%d'%i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    prediction = lasagne.layers.get_output(network, deterministic=True)
    result = T.argmax(prediction, axis=1)
    predict_fn = theano.function([input_var],result)
    return predict_fn


def pyramid(image, scale=1.25, minSize=(30, 30)):
    # yield the original image
    yield image
 
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
 
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
 
        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    y_range = []
    x_range = []
    i = 0
    j = 0
    while(i<=image.shape[0]):
        y_range.append(i)
        i = i+stepSize

    while(j<=image.shape[1]):
        x_range.append(j)
        j = j+stepSize

    for y in y_range:
        for x in x_range:
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


image = cv2.imread('test4.jpeg',0)
print(np.shape(image))
#image = cv2.resize(image,(28,28))
#print(np.shape(image))
(winW, winH) = (30,30)



for resized in pyramid(image, scale=1.25):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=21, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if (window.shape[0] == winH and window.shape[1] == winW):
            
            pred = predict_label(image=window)
            print(pred([[window]]))
           
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 255, 255), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
