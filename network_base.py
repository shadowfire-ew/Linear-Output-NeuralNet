"""network.py
contains an abstract base class for neural networks
also has global variables for function names
TODO: include the train and test net functions, as those seem agnostic...
TODO: expand the base class in case there are parts where the subclasses have the same functions
"""
import abc

class NeuralNet(abc.ABC):
    @abc.abstractmethod
    def Classify(self,inarray):
        raise NotImplementedError

LINEAR = 0
SIGMOID = 1