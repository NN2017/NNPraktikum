# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
import logistic_layer
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        input_dimension = self.trainingSet.input.shape[1]
        #hidden_dimension = 20
        output_dimension = 1

        # self.hiddenLayer = logistic_layer.LogisticLayer(
        #     input_dimension, hidden_dimension, activation="sigmoid",
        #     learningRate=learningRate, isClassifierLayer=False
        # )
        self.outLayer = logistic_layer.LogisticLayer(#hidden_dimension
            input_dimension, output_dimension, activation="sigmoid",
            learningRate=learningRate, isClassifierLayer=True
        )

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import BinaryCrossEntropyError, MeanSquaredError
        loss_bce = BinaryCrossEntropyError()
        loss_mse = MeanSquaredError()

        learned = False
        iteration = 0

        while not learned:
            totalError_mse = 0
            totalError_bce = 0
            # input has shape (784,) with grey values for the 28x28 pixels
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                input_biased = np.concatenate((np.array([1]),input))
                output = self.outLayer.forward(input_biased)
                totalError_mse += loss_mse.calculateError(label, output)
                totalError_bce += loss_bce.calculateError(label, output)

                # we only have one layer, and there is no weight/derivative on the output
                # so derivative of the Error on the output is just the difference
                # every output neuron should be weights equally here (and we only have one anyway)
                derivative_last_layer_mse = label - output
                # derivative_last_layer_bce = TODO: here the derivative of BCE should be inserted.
                # https://en.wikipedia.org/wiki/Binary_entropy_function#Derivative or the like
                self.outLayer.computeDerivative(derivative_last_layer_mse, np.array([1]))
                self.outLayer.updateWeights()

            totalError_bce = abs(totalError_bce)
            #totalError_mse = abs(totalError_mse)

            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error (MSE): %i; (BCE): %i", iteration, totalError_mse, totalError_bce)

            if totalError_mse == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.outLayer.forward(np.insert(testInstance,0,[1])) > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))