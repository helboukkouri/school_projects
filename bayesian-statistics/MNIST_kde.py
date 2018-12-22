# coding: utf-8

from __future__ import print_function

import os
import numpy as np
from scipy.stats import gaussian_kde

import keras
import keras.backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")


def make_scatterplots(i, j, model, x, n_epochs):
    compared_detectors = np.array([
                [prediction_class_i, prediction_class_j, predicted_class]
                for prediction_class_i, prediction_class_j, predicted_class
                in zip(model.predict(x)[:, i],
                       model.predict(x)[:, j],
                       model.predict_classes(x))
                if (predicted_class == i) | (predicted_class == j)
    ])

    colors = [
        "red" if v else "blue" for v in (compared_detectors[:, 2] == i)
    ]

    plt.figure(figsize=[10, 10])
    plt.scatter(compared_detectors[:, 0],
                compared_detectors[:, 1],
                facecolors="none", s=50, edgecolors=colors)
    min_lim = min(compared_detectors[:, 0].min(),
                  compared_detectors[:, 1].min()) - .05
    max_lim = max(compared_detectors[:, 0].max(),
                  compared_detectors[:, 1].max()) + .05
    plt.xlim((min_lim, max_lim))
    plt.ylim((min_lim, max_lim))
    plt.xlabel("network's output for number : {}".format(i))
    plt.ylabel("network's output for number : {}".format(j))
    plt.title("comparing outputs for numbers {} and {}".format(i, j))
    name = "compare_%d_%d_epoch_%d.png" % (i, j, n_epochs)
    plt.savefig(os.path.join("img", name))


def make_and_compile_model(num_classes):
    model = Sequential()
    model.add(Dense(256, activation='sigmoid', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="linear"))

    model.compile(
        loss='mean_squared_error',
        optimizer="adam", metrics=['accuracy']
    )
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
    plt.savefig(os.path.join("img", "model_accuracy_epoch_%d.png" % epochs))
    plt.clf()
    # summarize history for loss
    plt.figure(figsize=[10, 10])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss: MSE')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig(os.path.join("img", "model_loss_epoch_%d.png" % epochs))
    plt.clf()


def main():
    if not os.path.exists("img"):
        os.makedirs("img")

    # some parameters
    batch_size = 128
    num_classes = 10
    epochs = 1

    # the MNIST data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = 2 * keras.utils.to_categorical(y_train, num_classes) - 1
    y_test = 2 * keras.utils.to_categorical(y_test, num_classes) - 1

    # make and train a neural network on the MNIST data
    K.set_learning_phase(1)
    model = make_and_compile_model(num_classes)
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(x_test, y_test))

    plot_history(history, epochs)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the network's outputs on some couples of categories
    # in order to reproduce the papers figures
    print("Making scatterplots ...")
    for i, j in [(0, 1), (3, 4), (3, 5)]:
        make_scatterplots(i, j, model, x_test, n_epochs=epochs)

    # fit KDE on the ouputs and targets of the training data
    prediction = model.predict(x_train)
    target_categories = y_train.argmax(axis=1)

    kde = {}
    for category in set(target_categories):
        kde[category] = gaussian_kde(
            prediction[target_categories == category, :].T
        )

    def predict_proba(x):
        values = [kde[c](model.predict(x)) for c in range(10)]
        return [(values[c]/sum(values))[0] for c in range(10)]

    # plot the KDE-generated probabilities on a random test sample
    x = x_test[0:1]
    y = y_test[0].argmax()
    probas = predict_proba(x)
    sns.set_style("whitegrid")
    plt.figure(figsize=[10, 10])
    plt.bar(range(10), np.log(probas))
    plt.title("log-probabilities predicted using KDE for a "
              + "random test sample with target : {}".format(y))
    name = "proba_bar_plot.png"
    plt.xlabel("categories")
    plt.xticks(range(10))
    plt.ylabel("log-probabilities")
    plt.savefig(os.path.join("img", name))

    # fit KDE a restriction to only 2 categories and plot the densities
    Z = {}
    for i, j in [(0, 1), (3, 4), (3, 5)]:
        for k in [i, j]:
            values = np.vstack([
                prediction[target_categories == k][:, i],
                prediction[target_categories == k][:, j]
            ])
            kernel = gaussian_kde(values)
            X, Y = np.mgrid[-2:2:100j, -2:2:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z[i, j, k] = np.reshape(kernel(positions).T, X.shape)

    sns.set_style("white")
    for i, j in [(0, 1), (3, 4), (3, 5)]:
        p = prediction[
            (target_categories == i) | (target_categories == j)
        ]
        c = [
            "red" if v else "blue" for v in p[:, (i, j)].argmax(axis=1)
        ]

        fig, ax = plt.subplots(figsize=[10, 10])
        ax.imshow(np.rot90(Z[i, j, i] + Z[i, j, j]),
                  cmap=plt.cm.gist_earth_r, extent=[-2, 2, -2, 2])
        plt.scatter(p[:, i], p[:, j],
                    facecolors="none", alpha=.5,
                    edgecolors=c, s=1)
        min_lim = min(p[:, i].min(), p[:, j].min()) - .05
        max_lim = max(p[:, i].max(), p[:, j].max()) + .05
        plt.xlim((min_lim, max_lim))
        plt.ylim((min_lim, max_lim))
        plt.xlabel("network's output for number : {}".format(i))
        plt.ylabel("network's output for number : {}".format(j))
        plt.title("comparing outputs KDE for numbers {} and {} ".format(i, j))
        name = "compare_%d_%d_with_kde_epoch_%d.png" % (i, j, epochs)
        plt.savefig(os.path.join("img", name))


if __name__ == '__main__':
    main()
