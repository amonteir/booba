import matplotlib.pyplot as plt
import booba.src.dnn as boo
from booba.utils.planar_utils import load_moons_dataset, plot_decision_boundary


def main():
    """

    """

    # Load train and test datasets
    train_X, train_Y = load_moons_dataset()

    # 3-layer model
    n_x = train_X.shape[0]
    n_h1 = 5
    n_h2 = 2
    n_y = 1  # 4th / output layer
    layers_dims = [n_x, n_h1, n_h2, n_y]
    optimizer = "momentum"  # gd, momentum, adam

    # Create a new DNN model object
    dnn3layers = boo.DNNModel(layers_dims, initialization='relu_optimal', optimizer=optimizer)
    # Train the DNN

    dnn3layers.train_mini_batch(train_X, train_Y, learning_rate=0.0007, num_epochs=5000, print_cost=True,
                                print_every_x_iter=1000,
                                hyperparam_lambda=0,  # controls L2-regularization
                                keep_neurons_probability=1,
                                optimizer=optimizer,
                                decay=dnn3layers.schedule_lr_decay)  # controls dropout

    # Use the trained DNN to make predictions on datasets
    dnn3layers.predict(train_X, train_Y, dataset='train')

    # Plot decision boundary
    plot_title = "Model with {0} optimization".format(optimizer)
    plt.title(plot_title)
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: dnn3layers.predict_dec(x.T), train_X, train_Y)

    return


if __name__ == '__main__':
    main()
    print("\nGG and goodbye for now!")
