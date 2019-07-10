# Learning_Neural_Network

1. ## [BOOK](http://neuralnetworksanddeeplearning.com/chap1.html)
2. ### [the Gradient](https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/)
3. ### [Partial derivatives/ Gradient khan academy](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/introduction-to-partial-derivatives)


import numpy as np
class Network(object):

    def __init__(self, layersSizes):
        # layersSizes is the list of number of nodes in each layer including input, output
        # first one is input, last one is output, others are hidden layers
        self.num_layers = len(layersSizes)
        self.layersSizes = layersSizes
        self.biases = [np.random.randn(numberOfBayes, 1) for numberOfBayes in layersSizes[1:]]
        # Bayes needed to be start after the input layer so layersSizes[1:] is used
        # self.biases is a list of list like [[bayeses for layer 1], [bayeses for layer 2], .... , [bayeses for layer N-1]]
        # (numberOfBayes, 1) here 1 is for 1d array
        self.weights = [np.random.randn(row, col) 
                        for col, row in zip(layersSizes[:-1], layersSizes[1:])]
        # self.weights is a list of list like [[weights for layer 1], [weights for layer 2], .... , [weights for layer N-1]]
        # [:-1] means before last 1, [1:] means start from the second one
        
net = Network([2, 3, 1]) # 2 input nodes, 1 hidden layer with 3 nodes,  1 output node
