import matplotlib.pyplot as plt
import numpy as np
from booba.utils.dnn_utils import sigmoid, relu, \
    linear_forward, relu_backward, sigmoid_backward, \
    linear_backward

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
    def __init__(self, layers_dims, initialisation="he"):
        """
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
        """
        self.parameters = {}
        self.costs = []
        self.grads = {}
        self.layers_dims = layers_dims
        self.initialisation = initialisation
        self.initialise_parameters()

    def initialise_parameters(self):
        """
        Initialises the parameters of a DNN with a given number of layers and dimensions
        Initilisation is key in NN:
            Different initializations lead to very different results
            Random initialization is used to break symmetry and make sure different hidden units can
                learn different things
            Resist initializing to values that are too large!
            He initialization works well for networks with ReLU activations


        """
        np.random.seed(1)
        L = len(self.layers_dims)  # number of layers in the network

        match self.initialisation:
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
                                                        / np.sqrt(self.layers_dims[layer-1])
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

    def linear_activation_forward(self, A_prev, W, b, activation):
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

        if activation == "sigmoid":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        linear_activation_cache = (linear_cache, activation_cache)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        return A, linear_activation_cache

    def forward_prop(self, X):
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
                                                      self.parameters["b" + str(l)], "relu")
            caches.append(cache)

        # Implement last SIGMOID layer, LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, self.parameters["W" + str(L)],
                                                   self.parameters["b" + str(L)], "sigmoid")
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
        cross_entropy_cost_raw = (-1 / m) * np.sum(np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y.T)))
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

    def linear_activation_backward(self, dA, cache, activation, hyperparam_lambda):
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
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache, hyperparam_lambda)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache, hyperparam_lambda)

        return dA_prev, dW, db

    def backward_prop(self, AL, Y, caches, hyperparam_lambda):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        Arguments:
        AL -- probability vector, output of the forward propagation (forward_prop
())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
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
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, "sigmoid",
                                                                         hyperparam_lambda)
        self.grads["dA" + str(L - 1)] = dA_prev_temp
        self.grads["dW" + str(L)] = dW_temp
        self.grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, "relu",
                                                                             hyperparam_lambda)
            self.grads["dA" + str(l)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp

    def update_parameters(self, learning_rate):
        """
        Update parameters using gradient descent
        Arguments:
        params -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of backward_prop

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """
        L = len(self.parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - learning_rate * self.grads[
                "dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - learning_rate * self.grads[
                "db" + str(l + 1)]

    def train(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False, hyperparam_lambda=0.0,
              keep_prob=1):
        """

        :param X: input dataset
        :param Y: true labels
        :param learning_rate:
        :param num_iterations:
        :param print_cost: bool
        :param hyperparam_lambda: The value of lambd is a hyperparameter that you can tune using a dev set.
        L2 regularization makes your decision boundary smoother. If lambd
        is too large, it is also possible to "oversmooth", resulting in a model with high bias.
        :param keep_prob:
        :return:
        """

        # Gradient Descent loop
        for i in range(0, num_iterations):
            # Forward propagation: [LINEAR -> Relu_forward*(L-1) -> LINEAR -> SIGMOID
            if keep_prob == 1:
                AL, caches = self.forward_prop(X)
            elif keep_prob < 1:
                AL, caches = self.forward_prop_with_dropout(X, keep_prob)

            # Compute cost
            cost = self.compute_cost(AL, Y, hyperparam_lambda)

            # Backward propagation
            # output is to self.grads
            if keep_prob >= 1:
                self.backward_prop(AL, Y, caches, hyperparam_lambda)
            else:
                # self.backward_propagation_with_dropout(X, Y, caches, keep_prob)
                pass

            # Update parameters
            self.update_parameters(learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
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
        probas, caches = self.forward_prop(X)

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

    def print_mislabeled_images(self, classes, X, y, p):
        """
        Plots images where predictions and truth were different.
        X -- dataset
        y -- true labels
        p -- predictions
        """
        a = p + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
        num_images = len(mislabeled_indices[0])
        for i in range(num_images):
            index = mislabeled_indices[1][i]

            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[
                y[0, index]].decode("utf-8"))

        plt.show()