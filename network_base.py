"""network.py
contains an abstract base class for neural networks
"""
import abc

class NeuralNet(abc.ABC):
    @abc.abstractmethod
    def Classify(self,inarray):
        raise NotImplementedError