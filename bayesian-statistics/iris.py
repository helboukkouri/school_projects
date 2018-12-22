# coding: utf-8

import os
import numpy as np

import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")


def keras_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='sigmoid'))
    model.add(Dense(3, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['accuracy'])
    return model


def plot_history(history, epochs):
    # summarize history for accuracy
    sns.set_style("whitegrid")
    plt.figure(figsize=[10, 10])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig(os.path.join("img", "model_accuracy_epoch_%d_iris.png" % epochs))
    plt.clf()
    # summarize history for loss
    plt.figure(figsize=[10, 10])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss: MSE')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig(os.path.join("img", "model_loss_epoch_%d_iris.png" % epochs))
    plt.clf()


def compute_sigma(model, features, targets, delta=0.01, beta=1):
    """
    Computes the variance from the discretized gradients of a
    single training example's outputs with respect to the network's weights.
    """
    print("Computing gradients ...")

    original_output = model.predict(features)
    network_weights = [K.eval(w) for w in model.trainable_weights]
    sigmas = {}
    for c in set(targets):
        category_output = original_output[targets == c]
        category_features = features[targets == c]

        sigmas[c] = 0
        for idx, w in enumerate(network_weights):
            dim = len(w.shape)
            if dim > 1:
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        w[i, j] += delta
                        model.set_weights(network_weights)
                        output = model.predict(category_features)
                        gradient = (output - category_output) / delta
                        h = 2 * (gradient ** 2).sum()
                        gamma_2 = gradient ** 2
                        sigmas[c] += gamma_2/(beta * h)
                        w[i, j] += - delta

            elif dim == 1:
                for i in range(w.shape[0]):
                    w[i] += delta
                    model.set_weights(network_weights)
                    output = model.predict(category_features)
                    gradient = (output - category_output) / delta
                    h = 2 * gradient.sum()
                    gamma_2 = gradient ** 2
                    sigmas[c] += 2 * gamma_2/(beta * h)
                    w[i] += - delta
            else:
                raise Exception

        sigmas[c] = sigmas[c].mean(axis=0)

    return sigmas


def main():
    # load iris dataset
    iris = load_iris()
    data = iris['data']
    targets = iris['target']
    target_names = iris['target_names']

    # targets into -1, 1 one-hots
    y = 2 * to_categorical(targets, 3) - 1

    # split the data into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.33,
    )

    # make and compile a neural network
    model = keras_model()

    # train the network
    epochs = 500  # number of epochs
    history = model.fit(x_train, y_train,
                        batch_size=8, epochs=epochs,
                        validation_data=(x_test, y_test))

    # plot metrics and loss evolution
    plot_history(history, epochs)

    # computes the sigma squared of the paper's formula
    sigmas_squared = compute_sigma(
        model=model, features=x_train, targets=y_train.argmax(axis=1)
    )

    # print the results
    for c in range(3):
        print("category: %d" % c)
        print(sigmas_squared[c])

    # define a function that predict a probability
    # using the paper's formula
    def predict_proba(x):
        values = [
            np.exp(2 * model.predict(x)[0][c]/sigmas_squared[c][c])
            for c in range(3)
        ]
        return [(values[c]/sum(values)) for c in range(3)]

    # define the softmax for comparison
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    # sample a random example from the test set then compare the results
    idx = np.random.choice(range(x_test.shape[0]))
    x = x_test[idx:idx+1]
    y = y_test[idx].argmax()
    print("target: {}".format(y))
    print("network output: {}".format(model.predict(x)[0]))
    print("predicted proba (paper formula): {}".format(predict_proba(x)))
    print("predicted proba (softmax): {}".format(softmax(model.predict(x)[0])))

    # bar chart
    plt.figure(figsize=[10, 10])
    plt.bar(np.arange(3)-0.2, predict_proba(x), width=0.2, label="paper formula")
    plt.bar(np.arange(3), softmax(model.predict(x)[0]), width=0.2, label="softmax")
    plt.xlabel("category")
    plt.xticks(np.arange(3)-0.1, target_names)
    plt.ylabel("probability")
    plt.title("predicted probabilities using different preprocessors for a"
              + "random sample with target: {}".format(target_names[y]))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
