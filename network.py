"""
the main module that handles the neural networks
utilizes numpy for now
"""
import numpy as np

def LINEAR(inp):
    return inp

def SIGMOID(inp):
    return(1/(1+np.exp(-inp)))

def LINEAR_DERIVE(act):
    return act*0+1

def SIGMOID_DERIVE(act):
    return act*(1-act)

ACTIVATIONS = {LINEAR:LINEAR_DERIVE,SIGMOID:SIGMOID_DERIVE}

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
        if set(activations) != ACTIVATIONS.keys():
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
        return prop[-1]

    def ForwardProp(self,inarray):
        """
        performs forward propagation
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

    def BackProp(self,acts,label):
        """
        performs backpropagation
        takes the layer activations list from ForwardProp
        returns a delta_thetas list of matrices
        """
        label = np.array(label)
        if len(acts) != len(self._thetas)+1:
            raise Exception("Wrong amount of activations (expected {} got {})".format(len(self._thetas)+1,len(acts)))
        else:
            delta_thetas = [None]*len(self._thetas)
            deltas = [None]*len(self._thetas)
            for layer in range(-1,-len(acts),-1):
                # current layer's activation level
                this_act = acts[layer]
                # current layer's activation function
                this_func = self._funcs[layer]
                # partial derivative function of this function
                this_deriv_func = ACTIVATIONS[this_func]
                # the partial derivative of the activation function with respect to the activating iputs
                # because of way sigmoid and linear work, we don't actually need the inputs for the activation function
                # just the outputs are used as inputs for the derivative
                this_derive_vals = this_deriv_func(this_act)
                # calculating the derivative of the cost of this layer
                back_errr = 0
                if layer == -1:
                    # output layer, simple difference
                    back_errr = label-this_act
                else:
                    # hidden layers
                    # transpose theta to next
                    thetaT = np.transpose(self._thetas[layer+1])
                    # getting the next cost
                    cost_next = deltas[layer+1]
                    # bring cost back
                    back_errr = (np.matmul(cost_next,thetaT))[1:]
                    # don't care about cost of bias
                # saving that cost to the deltas list
                cost_derive = back_errr*this_derive_vals
                deltas[layer] = cost_derive
                # create the delta_theta part
                # get previous activation & bias
                preact = np.concatenate(([1],acts[layer-1]))
                # transpose activation to vert
                actT = preact.reshape((len(preact),1))
                # reshape derivative
                deriM = cost_derive.reshape((1,len(cost_derive)))
                #matmul to make delta_theta
                delta_theta = np.matmul(actT,deriM)
                # save it to delta thetas
                delta_thetas[layer] = delta_theta
        return delta_thetas


if __name__ == "__main__":
    testNN = NeuralNet(5,20,[7,6,10],[SIGMOID,LINEAR,SIGMOID,LINEAR])
    for theta in testNN._thetas:
        print(theta.shape)
    for act in testNN.ForwardProp([1,2,3,4,5]):
        print(act)