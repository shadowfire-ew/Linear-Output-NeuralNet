"""netwprk.py
the main module that handles the neural networks
utilizes numpy for now
"""
import numpy as np
import time

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

    def Cost(self,inp,label):
        """
        applies squared dif cost to one classification
        """
        label = np.array(label)
        guess= self.Classify(inp)
        costs = (guess-label)
        costs2 = costs*costs
        costsum = sum(costs2)
        return costsum/len(guess)
    
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

    def ApplyDeltaTheta(self,delta_thetas,alpha=0.05):
        """
        takes the delta_thetas from Backprop
        and uses them to inform changes to current thetas
        can also be an average of many delta_thetas
        alpha is learning rate
        """
        if len(delta_thetas) != len(self._thetas):
            raise Exception("Wroung amount of delta theta (expected {} got {})".format(len(self._theta),len(delta_thetas)))
        else:
            for i in range(len(delta_thetas)):
                self._thetas[i] += alpha*delta_thetas[i]

def TrainNet(net,data,epochs,alpha=0.05):
    """
    trains the neural net on the data
    assumes type net is NeuralNet class
        and data is list of features-labels pairs
    epochs is number of times dataset is learned
    alpha is learning rate
    """
    start = time.perf_counter()
    # validate net
    if not isinstance(net,NeuralNet):
        raise TypeError("net input is not an instance of NeuralNet class")
    # validate data
    pinlen,poutlen = None,None
    for x in data:
        if len(x) != 2:
            raise Exception("One datapoint is not a pair:",x)
        if pinlen is not None and len(x[0]) != pinlen:
            raise Exception("One datapoint has anomalous input:",x)
        else:
            pinlen = len(x[0])
        if poutlen is not None and len(x[1]) != poutlen:
            raise Exception("One datapoint has anomolous output:",x)
        else:
            poutlen = len(x[1])
    # validate epochs
    if type(epochs) is not int or 0 < epochs:
        raise ValueError("Cannot run {} epochs".format(epochs))
    # validate alpha
    if type(alpha) is not int and type(alpha) is not float:
        raise ValueError("Unexpected value for alpha:",alpha)
    # if all tests pass, input valid
    # itterate through epochs
    print("Begining training")
    for epoch in range(epochs):
        # print a reminder every 10% of the way
        if epoch%(epochs//10) == 0 and epoch != 0:
            print("Starting epoch {}...".format(epoch))
        for e in data:
            x = e[0]
            y = e[1]
            acts = net.ForwardProp(x)
            dthetas = net.BackProp(acts,y)
            net.ApplyDeltaTheta(dthetas,alpha)
    # after all epochs
    done = time.perf_counter()-start
    print("Done in {m} minutes and {s:.2f} seconds".format(m=done//60,s=done%60))

def TestNet(net,data):
    """
    runs the net on the test data
    and returns the average cost
    """
    # validate net
    if not isinstance(net,NeuralNet):
        raise TypeError("net input is not an instance of NeuralNet class")
    # validate data
    pinlen,poutlen = None,None
    for x in data:
        if len(x) != 2:
            raise Exception("One datapoint is not a pair:",x)
        if pinlen is not None and len(x[0]) != pinlen:
            raise Exception("One datapoint has anomalous input:",x)
        else:
            pinlen = len(x[0])
        if poutlen is not None and len(x[1]) != poutlen:
            raise Exception("One datapoint has anomolous output:",x)
        else:
            poutlen = len(x[1])
    # validation complete
    # start summing costs
    sumpart = 0
    for e in data:
        x = e[0]
        y = e[1]
        sumpart += net.Cost(x,y)
    # return average
    return sumpart/len(data)

if __name__ == "__main__":
    x = [1,2,3,4,5]
    y = [5,7]
    testNN = NeuralNet(len(x),len(y),[7,6],[SIGMOID,SIGMOID,LINEAR])
    print("thetas")
    for theta in testNN._thetas:
        print(theta.shape)
    acts = testNN.ForwardProp(x)
    print("activations")
    for act in acts:
        print(act)
    print("applying backprop")
    dthetas = testNN.BackProp(acts,y)
    print("Delta thetas")
    for dtheta in dthetas:
        print(dtheta)
    print("Applying delta_thetas")
    print("cost before:",testNN.Cost(x,y))
    testNN.ApplyDeltaTheta(dthetas)
    print("cost after:",testNN.Cost(x,y))