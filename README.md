# Learning_Neural_Network

1. ## [BOOK](http://neuralnetworksanddeeplearning.com/chap1.html)
2. ### [the Gradient](https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/)
3. ### [Partial derivatives/ Gradient khan academy](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/introduction-to-partial-derivatives)


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        # range(start, stop, step)
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch ",j, ": ",self.evaluate(test_data), " / ", n_test)
            else:
                print("Epoch complete:", j)
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def feedforward(self, x):
       
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
        return x
        """
        Return the output of the network if "x" is input.
        input x can be x1, x2, ..... xn
        """
        """
        It is assumed that the input x is an (n, 1) Numpy ndarray, not x (n,) vector.
        Here, n is the number of inputs to the network. If you try to use an (n,) vector as 
        input you'll get strange results. Although using an (n,) vector appears the more natural choice, 
        using an (n, 1) ndarray makes it particularly easy to modify the code to feedforward
        multiple inputs at once, and that is sometimes convenient.
        """
