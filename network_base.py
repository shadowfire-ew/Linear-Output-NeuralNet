"""network.py
contains an abstract base class for neural networks
also has global variables for function names
"""
import abc

class NeuralNet(abc.ABC):
    @abc.abstractmethod
    def Classify(self,inarray):
        raise NotImplementedError

LINEAR = 0
SIGMOID = 1