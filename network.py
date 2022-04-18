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
    
    def Classify(self,inarray):
        """
        returns the classification of the input array
        """
        prop = self.ForwardProp(inarray)
        return prop[0]

    def ForwardProp(self,inarray):
        """
        performs forward propogation
        returns a list of all layer activations
        this includes the input layer
        """
        #TODO: research if bias is actually needed
        #       or changes when linear layers are considered
        # setting up the activations list, primed with input
        acts = [inarray]
        for layer in range(len(self._thetas)):
            # get previous activation and apply bias
            p_layer = np.concatenate(([1],acts[-1]))
            # apply transformation
            n_array = np.matmul(p_layer,self._thetas[layer])
            # apply activation function
            activated = self._funcs[layer](n_array)
            # save array to activations
            acts.append(activated)
        return acts

if __name__ == "__main__":
    testNN = NeuralNet(5,20,[7,6],[SIGMOID,SIGMOID,LINEAR])
    for theta in testNN._thetas:
        print(theta.shape)