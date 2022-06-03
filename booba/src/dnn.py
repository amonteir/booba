import math

import matplotlib.pyplot as plt
import numpy as np
from booba.utils.dnn_utils import sigmoid, relu, \
    linear_forward, relu_backward, sigmoid_backward, \
    linear_backward, linear_forward_dropout

"""
Step1. Initialize the parameters for a two-layer network and for an 𝐿-layer neural network
Step2. Implement the forward propagation module (shown in purple in the figure below)
        - Complete the LINEAR part of a layer's forward propagation step (resulting in 𝑍[𝑙]).
        - The ACTIVATION function is provided for you (relu/sigmoid)
        - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
        - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add
            a [LINEAR->SIGMOID] at the end (for the final layer 𝐿). This gives you a new forward_prop
 function.
Step3. Compute the loss
Step4. Implement the backward propagation module (denoted in red in the figure below)
        - Complete the LINEAR part of a layer's backward propagation step
        - The gradient of the ACTIVATE function is provided for you(relu_backward/sigmoid_backward)
        - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function
        - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new backward_prop
 function
Step5. Finally, update the parameters
"""


class DNNModel:
    def __init__(self, layers_dims, initialization="he", optimizer="gd"):
        """
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
        """
        self.parameters = {}
        self.costs = []
        self.grads = {}
        self.velocity = {}
        self.squared_grad = {}
        self.layers_dims = layers_dims
        self.initialization = initialization
        self.initialize_parameters()

        self.optimizers_set = set()
        self.optimizers_set.update(['momentum', 'adam', 'gd'])
        if optimizer in self.optimizers_set:
            if optimizer == 'gd':
                pass
            elif optimizer == 'momentum':
                self.initialize_velocity()
            elif optimizer == 'adam':
                self.initialize_adam()
        else:
            return

    def initialize_velocity(self):
        """
        Initializes the velocity to be used with Momentum:
                    - keys: "dW1", "db1", ..., "dWL", "dbL"
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        """

        L = len(self.layers_dims)  # number of layers in the neural network

        # Initialize velocity
        for l in range(1, L + 1):
            self.velocity["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            self.velocity["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)

    def initialize_adam(self):
        """
        Initializes velocity and squared gradient as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL"
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        """

        L = len(self.parameters) // 2  # number of layers in the neural networks

        # Initialize velocity and squared_grad
        for l in range(L):
            self.velocity["dW" + str(l + 1)] = np.zeros(self.parameters["W" + str(l + 1)].shape)
            self.velocity["db" + str(l + 1)] = np.zeros(self.parameters["b" + str(l + 1)].shape)
            self.squared_grad["dW" + str(l + 1)] = np.zeros(self.parameters["W" + str(l + 1)].shape)
            self.squared_grad["db" + str(l + 1)] = np.zeros(self.parameters["b" + str(l + 1)].shape)

    def initialize_parameters(self):
        """
        Initialises the parameters of a DNN with a given number of layers and dimensions
        Initilisation is key in NN:
            Different initializations lead to very different results
            Random initialization is used to break symmetry and make sure different hidden units can
                learn different things
            Resist initializing to values that are too large!
            He initialization works well for networks with ReLU activations


        """
        np.random.seed(3)
        L = len(self.layers_dims)  # number of layers in the network

        match self.initialization:
            case "zeros":
                """
                Zeros Initialisation: very bad idea!In general, initializing all the weights to zero 
                results in the network failing to break symmetry.
                The weights 𝑊[𝑙] should be initialized randomly to break symmetry.
                However, it's okay to initialize the biases 𝑏[𝑙] to zeros. 
                Symmetry is still broken so long as 𝑊[𝑙] is initialized randomly. 
                """
                for layer in range(1, L):
                    self.parameters["W" + str(layer)] = np.zeros((self.layers_dims[layer], self.layers_dims[layer - 1]))
                    self.parameters["b" + str(layer)] = np.zeros((self.layers_dims[layer], 1))
                    assert (self.parameters['W' + str(layer)].shape == (self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]))
                    assert (self.parameters['b' + str(layer)].shape == (self.layers_dims[layer], 1))

            case "he":
                for layer in range(1, L):
                    self.parameters["W" + str(layer)] = np.random.randn(self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]) * \
                                                        np.sqrt(2. / self.layers_dims[layer - 1])
                    self.parameters["b" + str(layer)] = np.zeros((self.layers_dims[layer], 1))
                    assert (self.parameters['W' + str(layer)].shape == (self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]))
                    assert (self.parameters['b' + str(layer)].shape == (self.layers_dims[layer], 1))

            case "xavier":
                for layer in range(1, L):
                    self.parameters["W" + str(layer)] = np.random.randn(self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]) * \
                                                        np.sqrt(1. / self.layers_dims[layer - 1])
                    self.parameters["b" + str(layer)] = np.zeros((self.layers_dims[layer], 1))
                    assert (self.parameters['W' + str(layer)].shape == (self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]))
                    assert (self.parameters['b' + str(layer)].shape == (self.layers_dims[layer], 1))

            case "random":
                """
                With large random-valued weights (... * 10 or more), 
                the last activation (sigmoid) outputs results that are very close to 0 or 1 
                for some examples, and when it gets that example wrong it incurs a very 
                high loss for that example. Indeed, when log(𝑎[3])=log(0), the loss goes to infinity.
                Hence why * 0.01
                """
                for layer in range(1, L):
                    self.parameters["W" + str(layer)] = np.random.randn(self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]) * 0.01
                    self.parameters["b" + str(layer)] = np.zeros((self.layers_dims[layer], 1))
                    assert (self.parameters['W' + str(layer)].shape == (self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]))
                    assert (self.parameters['b' + str(layer)].shape == (self.layers_dims[layer], 1))

            case "relu_optimal":
                for layer in range(1, L):
                    self.parameters["W" + str(layer)] = np.random.randn(self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]) \
                                                        / np.sqrt(self.layers_dims[layer - 1])
                    self.parameters["b" + str(layer)] = np.zeros((self.layers_dims[layer], 1))
                    assert (self.parameters['W' + str(layer)].shape == (self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]))
                    assert (self.parameters['b' + str(layer)].shape == (self.layers_dims[layer], 1))

            case _:
                for layer in range(1, L):
                    self.parameters["W" + str(layer)] = np.random.randn(self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]) * 0.01
                    self.parameters["b" + str(layer)] = np.zeros((self.layers_dims[layer], 1))
                    assert (self.parameters['W' + str(layer)].shape == (self.layers_dims[layer],
                                                                        self.layers_dims[layer - 1]))
                    assert (self.parameters['b' + str(layer)].shape == (self.layers_dims[layer], 1))

    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation, keep_neurons_probability):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
        Arguments:
            A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples1)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        Returns:
            A -- the output of the activation function, also called the post-activation value
            cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        linear_activation_cache = []

        if activation == "sigmoid":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
            dummy = np.full((A.shape[0], A.shape[1]), -1)
            linear_activation_cache = (linear_cache, activation_cache, dummy)


        elif activation == "relu":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            if 1 > keep_neurons_probability > 0:
                # apply Dropout
                A, dropout_cache = linear_forward_dropout(A, keep_neurons_probability)
                linear_activation_cache = (linear_cache, activation_cache, dropout_cache)
            else:
                dummy = np.full((A.shape[0], A.shape[1]), -1)
                linear_activation_cache = (linear_cache, activation_cache, dummy)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        return A, linear_activation_cache

    def forward_prop(self, X, keep_neurons_probability):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        Arguments:
        X -- data, numpy array of shape (input size, number of examples1)
        parameters -- output of initialize_parameters_deep()
        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """
        A = X
        L = len(self.parameters) // 2  # number of layers in the neural network
        caches = []

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        # The for loop starts at 1 because layer 0 is the input
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, self.parameters["W" + str(l)],
                                                      self.parameters["b" + str(l)], "relu",
                                                      keep_neurons_probability)
            caches.append(cache)

        # Implement last SIGMOID layer, LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, self.parameters["W" + str(L)],
                                                   self.parameters["b" + str(L)], "sigmoid",
                                                   keep_neurons_probability)
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def compute_cost(self, AL, Y, hyperparam_lambda):
        """
        Implement the cost function
        Arguments:
        AL -- probability vector corresponding to your label predictions
        Y -- true "label" vector
        hyperparam_lambda - regularization hyperparameter lambda
        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        L = len(self.layers_dims)  # number of layers in the network
        assert (hyperparam_lambda >= 0.0)

        # Compute cross entropy cost
        # Logprobs = np.multiply(np.log(AL), Y) + np.multiply(1 - Y, np.log(1 - AL))
        # cost = (-1 / m) * np.sum(Logprobs)
        # cross_entropy_cost_raw = (-1 / m) * np.sum(np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y.T)))
        AL = np.where(AL >= 1, 1 - 1E-6, AL)
        cross_entropy_cost_raw = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cross_entropy_cost = np.squeeze(cross_entropy_cost_raw)  # turns [[17]] into 17)

        L2_regularization_cost = 0.0

        if hyperparam_lambda > 0.0:
            # Regularization part of the cost
            sums = 0
            for layer in range(1, L):
                sums += np.sum(np.square(self.parameters["W" + str(layer)]))
            L2_regularization_cost = (1 / m) * (hyperparam_lambda / 2) * sums

        # Add both costs
        cost = cross_entropy_cost + L2_regularization_cost
        assert (cost.shape == ())

        return cost

    @staticmethod
    def linear_activation_backward(dA, cache, activation, hyperparam_lambda):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        linear_cache, activation_cache, _ = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache, hyperparam_lambda)

        return dA_prev, dW, db

    def backward_prop(self, AL, Y, caches, hyperparam_lambda, keep_neurons_probability):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        Arguments:
        AL -- probability vector, output of the forward propagation (forward_prop())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l],
                    for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        AL = np.where(AL >= 1, 1 - 1E-6, AL)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

        # Lth layer (SIGMOID -> LINEAR) gradients.
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache,
                                                                         "sigmoid",
                                                                         hyperparam_lambda)
        if 1 > keep_neurons_probability > 0:
            # apply dropout
            _, _, dropout_cache = caches[L - 2]
            dA_prev_temp = dA_prev_temp * dropout_cache
            dA_prev_temp = dA_prev_temp / keep_neurons_probability

        self.grads["dA" + str(L - 1)] = dA_prev_temp
        self.grads["dW" + str(L)] = dW_temp
        self.grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache,
                                                                             "relu",
                                                                             hyperparam_lambda)
            if 1 > keep_neurons_probability > 0 and l > 0:
                # apply dropout
                _, _, dropout_cache = caches[l - 1]
                dA_prev_temp = dA_prev_temp * dropout_cache
                dA_prev_temp = dA_prev_temp / keep_neurons_probability

            self.grads["dA" + str(l)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp

    def update_parameters(self, learning_rate):
        """
        Update parameters using gradient descent
        """
        L = len(self.parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - learning_rate * self.grads[
                "dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - learning_rate * self.grads[
                "db" + str(l + 1)]

    def update_parameters_with_momentum(self, learning_rate, beta=0.9):
        """
        Update parameters using gradient descent with momentum
        """
        L = len(self.parameters) // 2  # number of layers in the neural network

        for l in range(L):
            # beta = 0, means no momentum
            self.velocity["dW" + str(l + 1)] = beta * self.velocity["dW" + str(l + 1)] + (1 - beta) \
                                               * self.grads['dW' + str(l + 1)]
            self.velocity["db" + str(l + 1)] = beta * self.velocity["db" + str(l + 1)] + (1 - beta) \
                                               * self.grads['db' + str(l + 1)]
            # update parameters
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - \
                                                learning_rate * self.velocity["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - \
                                                learning_rate * self.velocity["db" + str(l + 1)]


    def update_parameters_with_adam(self, t, learning_rate=0.01,
                                    beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Update parameters using Adam
        beta1 -- Exponential decay hyperparameter for the first moment estimates
        beta2 -- Exponential decay hyperparameter for the second moment estimates
        epsilon -- hyperparameter preventing division by zero in Adam updates
        """

        L = len(self.parameters) // 2  # number of layers in the neural networks
        v_corrected = {}  # Initializing first moment estimate, python dictionary
        s_corrected = {}  # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients.
            self.velocity["dW" + str(l + 1)] = beta1 * self.velocity["dW" + str(l + 1)] + \
                                               (1 - beta1) * self.grads['dW' + str(l + 1)]
            self.velocity["db" + str(l + 1)] = beta1 * self.velocity["db" + str(l + 1)] + \
                                               (1 - beta1) * self.grads['db' + str(l + 1)]

            # Compute bias-corrected first moment estimate.
            v_corrected["dW" + str(l + 1)] = self.velocity["dW" + str(l + 1)] / (1 - beta1 ** t)
            v_corrected["db" + str(l + 1)] = self.velocity["db" + str(l + 1)] / (1 - beta1 ** t)

            # Moving average of the squared gradients.
            self.squared_grad["dW" + str(l + 1)] = beta2 * self.squared_grad["dW" + str(l + 1)] + \
                                                   (1 - beta2) * (self.grads['dW' + str(l + 1)]) ** 2
            self.squared_grad["db" + str(l + 1)] = beta2 * self.squared_grad["db" + str(l + 1)] + \
                                                   (1 - beta2) * (self.grads['db' + str(l + 1)]) ** 2

            # Compute bias-corrected second raw moment estimate.
            s_corrected["dW" + str(l + 1)] = self.squared_grad["dW" + str(l + 1)] / (1 - beta2 ** t)
            s_corrected["db" + str(l + 1)] = self.squared_grad["db" + str(l + 1)] / (1 - beta2 ** t)

            # Update parameters
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - \
                                                (learning_rate * v_corrected["dW" + str(l + 1)]) / (
                                                    np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - \
                                                (learning_rate * v_corrected["db" + str(l + 1)]) / (
                                                    np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

    def train(self, X, Y, learning_rate=0.0075, num_epochs=10000, print_cost=False,
              print_every_x_iter=1000, hyperparam_lambda=0.0, keep_neurons_probability=1,
              optimizer='adam', mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
              epsilon=1e-8):
        """
        Train the model
        """
        print('\nStart training model.')

        # Gradient Descent loop
        for i in range(0, num_epochs):
            # Forward propagation: [LINEAR -> Relu_forward*(L-1) -> LINEAR -> SIGMOID
            AL, caches = self.forward_prop(X, keep_neurons_probability)

            # Compute cost
            cost = self.compute_cost(AL, Y, hyperparam_lambda)

            # Backward propagation
            self.backward_prop(AL, Y, caches, hyperparam_lambda, keep_neurons_probability)

            # Update parameters
            self.update_parameters(learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % print_every_x_iter == 0:
                print("Iteration %i ::: Cost %f" % (i, cost))
            if print_cost and i % 100 == 0:
                self.costs.append(cost)

        # End Gradient Descent loop

        # plot the cost
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    def predict(self, X, y, dataset):
        """
        This function is used to predict the results of a L-layer neural network.

        Arguments:
        X -- data set of examples1 you would like to label
        y -- vector with true labels

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(self.parameters) // 2  # number of layers in the neural network
        predictions = np.zeros((1, m))

        # Forward propagation
        probas, caches = self.forward_prop(X, keep_neurons_probability=1)  # dropout must not be used in testing

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        print("Prediction accuracy on {} dataset: {}".format(dataset, str(np.sum((predictions == y) / m))))

        return predictions

    def predict_dec(self, X):
        """
        Used for plotting decision boundary.

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (m, K)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Predict using forward propagation and a classification threshold of 0.5
        a3, cache = self.forward_prop(X, keep_neurons_probability=1)
        predictions = (a3 > 0.5)
        return predictions

    @staticmethod
    def print_mislabeled_images(classes, X, y, p, num_images):
        """
        Plots images where predictions and truth were different.
        X -- dataset
        y -- true labels
        p -- predictions
        """
        a = p + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
        # num_images = len(mislabeled_indices[0])

        for i in range(num_images):
            index = mislabeled_indices[1][i]

            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[
                y[0, index]].decode("utf-8"))

        plt.show()

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        np.random.seed(seed)  # To make your "random" minibatches the same as ours
        m = X.shape[1]  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        # Step 2 - Partition (shuffled_X, shuffled_Y).
        # Cases with a complete mini batch size only i.e each of 64 examples.
        num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size
        # in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches
