"""
the main module that handles the neural networks
utilizes numpy for now
"""
import numpy as np

def LINEAR(inp):
    return inp

def SIGMOID(inp):
    return(1/(1+np.exp(-inp)))

ACTIVATIONS = {LINEAR,SIGMOID}

class NeuralNet:
    """
    the base class for neural networks
    """
    def __init__(self,inputs,outputs,hiddens=[],activations=[LINEAR]):
        """
        initializes Neural Network class
        inputs is an integer determining the size of the input array
        outputs is likewise an integer for ths size of the output array
        hiddens is an array of integers where each represents the size of the activation array at that point
        activations is an array of functions
        """
