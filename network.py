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
