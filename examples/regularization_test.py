import matplotlib.pyplot as plt
import booba.src.dnn as boo
from booba.utils.planar_utils import load_2D_dataset, plot_decision_boundary


def main():
    """
    Each dot corresponds to a position on the football field where a football player has hit the ball
    with his/her head after the French goalkeeper has shot the ball from the left side of the football field.

    If the dot is blue, it means the French player managed to hit the ball with his/her head
    If the dot is red, it means the other team's player hit the ball with their head

    Goal: Use a deep learning model to find the positions on the field where the goalkeeper should kick the ball.

    """

    # Load train and test datasets
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    # 3-layer model
    n_x = train_X.shape[0]
    n_h1 = 20
    n_h2 = 3
    n_y = 1  # 4th / output layer
    layers_dims = [n_x, n_h1, n_h2, n_y]

    ### NO REGULARIZATION, NO DROPOUT
    # Create a new DNN model object
    dnn3layers = boo.DNNModel(layers_dims, initialization='relu_optimal')
    # Train the DNN without regularization
    dnn3layers.train(train_X, train_Y, learning_rate=0.3, num_epochs=30000, print_cost=True,
                     print_every_x_iter=10000,
                     hyperparam_lambda=0,  # controls L2-regularization
                     keep_neurons_probability=1,
                     optimizer='gd')  # controls dropout

    # Use the trained DNN to make predictions on datasets
    dnn3layers.predict(train_X, train_Y, dataset='train')
    predictions_test = dnn3layers.predict(test_X, test_Y, dataset='test')

    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: dnn3layers.predict_dec(x.T), train_X, train_Y)

    ### APPLY L2-REGULARIZATION, NO DROPOUT
    # Create a new DNN model object
    dnn3layers_reg = boo.DNNModel(layers_dims, initialization='relu_optimal')
    # Train the DNN without regularization
    dnn3layers_reg.train(train_X, train_Y, learning_rate=0.3, num_epochs=30000, print_cost=True,
                         print_every_x_iter=10000,
                         hyperparam_lambda=0.7,  # controls L2-regularization
                         keep_neurons_probability=1,
                         optimizer='gd')  # controls dropout

    # Use the trained DNN to make predictions on datasets
    dnn3layers_reg.predict(train_X, train_Y, dataset='train')
    predictions_test = dnn3layers_reg.predict(test_X, test_Y, dataset='test')

    plt.title("Model with L2-Regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: dnn3layers_reg.predict_dec(x.T), train_X, train_Y)

    ### APPLY DROPOUT, NO L2-REGULARIZATION
    # Create a new DNN model object
    dnn3layers_dropout = boo.DNNModel(layers_dims, initialization='relu_optimal')
    # Train the DNN without regularization
    dnn3layers_dropout.train(train_X, train_Y, learning_rate=0.3, num_epochs=50000, print_cost=True,
                             print_every_x_iter=10000,
                             hyperparam_lambda=0,  # controls L2-regularization
                             keep_neurons_probability=0.86,
                             optimizer='gd')  # controls dropout

    # Use the trained DNN to make predictions on datasets
    dnn3layers_dropout.predict(train_X, train_Y, dataset='train')
    predictions_test = dnn3layers_dropout.predict(test_X, test_Y, dataset='test')

    plt.title("Model with Dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: dnn3layers_dropout.predict_dec(x.T), train_X, train_Y)

    return


if __name__ == '__main__':
    main()
    print("\nGG and goodbye for now!")
