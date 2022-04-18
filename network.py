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
        # input validation section

        # Making sure the activations match with the hidden layers & output
        if len(activations) != len(hiddens)+1:
            raise Exception("Mismatching layers count and activation functions expected. #hiddens+output:{a} != #activations:{b}".format(a=len(hiddens)+1,b=len(activations)))
        
        # making sure activation functions are recognized
        if set(activations) != ACTIVATIONS:
            raise Exception("Unrecognized function in activations list")

        #checking for integer values for inputs, outputs, and hiddens
        if type(inputs) is not int:
            raise TypeError("Expected integer value for inputs")
        if type(outputs) is not int:
            raise TypeError("Expected integer value for outputs")
        for h in hiddens:
            if type(h) is not int:
                raise TypeError("Expected integer value for hidden layers")
        
        # after all checks passed, can save information we need

        # save our activations
        self._funcs = activations

        # begin construction of 
        # our layer transformations
        layers = hiddens+[outputs]
        # the previous layer array len
        prev = inputs
        # the actual matrices used for transformation
        self._thetas = []
        for nex in layers:
            # creating the theta matrix for this latyer
            # prev+1 is to account for bias
            self._thetas.append(np.random.rand(prev+1,nex)-0.5)
            # 
            prev = nex

if __name__ == "__main__":
    testNN = NeuralNet(5,20,[7,6],[SIGMOID,SIGMOID,LINEAR])
    for theta in testNN._thetas:
        print(theta.shape)