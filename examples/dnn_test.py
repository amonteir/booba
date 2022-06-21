import booba.src.dnn as boo
from booba.utils.process_datasets import load_data


def main():
    """
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    """

    # Load train and test datasets
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    """
    # Explore the dataset
    index = 10
    plt.imshow(train_x_orig[index])
    print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
    plt.show()
    
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    
    print("Number of training examples1: " + str(m_train))
    print("Number of testing examples1: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))
    """

    # Reshape the training and test examples1
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    """
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    """

    # 4-layer model
    n_x = 12288  # num_px * num_px * 3
    n_h1 = 20
    n_h2 = 7
    n_h3 = 5
    n_y = 1  # 4th / output layer
    layers_dims = [n_x, n_h1, n_h2, n_h3, n_y]

    # Create a new DNN model object
    dnn4layers = boo.DNNModel(layers_dims, initialization='relu_optimal')
    # Train the DNN

    dnn4layers.train(train_x, train_y, learning_rate=0.0075, num_epochs=1000, print_cost=True,
                     print_every_x_iter=100,
                     hyperparam_lambda=0.7,
                     keep_neurons_probability=1)


    # Use the trained DNN to make predictions on datasets
    dnn4layers.predict(train_x, train_y, dataset='train')
    predictions_test = dnn4layers.predict(test_x, test_y, dataset='test')

    dnn4layers.print_mislabeled_images(classes, test_x, test_y, predictions_test, 5)

    return


if __name__ == '__main__':
    main()
    print("\nGG and goodbye for now!")
