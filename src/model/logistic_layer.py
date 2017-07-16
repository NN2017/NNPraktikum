
import time

import numpy as np

from util.activation_functions import Activation
#from model.layer import Layer


class LogisticLayer:#(Layer):
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True, learningRate=0.05):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activation_derivative = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        #self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        self.learningRate = learningRate

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer
            input[0] is always 1 and represents the bias.

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer

        """

        self.input = input

        # (nOut,1)                      (nOut, nIn+1)  (1,nIn+1)
        self.before_activation = np.matmul(self.weights, input)
        self.output = self.activation(self.before_activation)
        return self.output

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        self.delta = self.activation_derivative(self.before_activation) * \
            nextDerivatives.dot(nextWeights)


    def updateWeights(self):
        """
        Update the weights of the layer
        weights matrix has shape (1,28^2), only one output but 28x28 (flattened) input
        """
        for i in range(self.nOut):  # loop over all neurons (here only one)
            for j in range(self.nIn+1): # loop over the input connections to each neuron (each also has a bias)
                self.weights[i,j] += self.delta[i] * self.input[j] * self.learningRate
