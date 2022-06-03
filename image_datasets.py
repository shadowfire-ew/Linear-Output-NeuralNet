"""image_datasets.py
a module to make loading of images into a dataset easier
"""
from PIL import Image
import random
import numpy as np
import network_base as nb

def ReadImage(path,shuffle = False):
    """
    Reads the image and returns a list pairing x,y coordinates as features and color channels as labels
    x and y coordinates are normalized along shape
    channel values are normalized by dividing by 255 
    """
    # read the image data as an array
    with Image.open(path) as imobj:
        imarr = np.asarray(imobj)
    # empty dataset
    dataset = []
    # traverse image
    for x in range(imarr.shape[0]):
        for y in range(imarr.shape[1]):
            # normalize coordinates based on shape
            xnorm = x/imarr.shape[0]
            ynorm = y/imarr.shape[1]
            # pair them up
            coords = (xnorm,ynorm)
            # normalize color channels (format agnostic)
            color = []
            for chan in range(imarr.shape[2]):
                # does assume that the values wll be integers from 0-255
                color.append(imarr[x][y][chan]/255)
            # convert to tuple (Why? Just standardizing? Implying that we do not expect this to change?)
            color=tuple(color)
            # make a feature-label pair
            pair = (coords,color)
            # add pair to end of list
            dataset.append(pair)
    if shuffle:
        random.shuffle(dataset)
    # return the datset as well as the original image shape (this will be useful in other places)
    return dataset,imarr.shape

def NetImage(net,shape):
    """
    a function to reconstruct an 
    """
    if not isinstance(net,nb.NeuralNetwork):
        raise TypeError("Unexpected type")
    # making a blank canvas to draw on
    newimarr = np.zeros(shape,dtype=np.uint8)
    for x in range(shape[0]):
        for y in range(shape[1]):
            # normalizing coords
            xnorm = x/shape[0]
            ynorm = y/shape[1]
            # pairing coords
            coords = (xnorm,ynorm)
            # getting color classification from the net
            color = net.Classify(coords)
            # applying colors to canvas (one chanel at a time)
            for chan in range(shape[2]):
                # base color
                chancol = 0
                try:
                    # get integer representation of that color
                    chancol = round(color[chan]*255)
                except:
                    # had issues with previous line involving inf/nan
                    # using this to just ignore casting issues
                    chancol = 0
                # squishing the colors into acceptable ranges
                # uses floor and ceiling
                # potential idea: overflow looping (%255) or reversing (advanced calc)
                if chancol < 0:
                    chancol = 0
                elif chancol > 255:
                    chancol = 255
                # painting the canvas
                newimarr[x][y][chan] = chancol
    return Image.fromarray(newimarr)