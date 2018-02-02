"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os
mpl.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer


FLAGS = flags.FLAGS

"""
Neural network is trained to create adversarial examples, for a black box
attack on MNIST dataset.
"""
# Up-sampling 2-D Layer (deconvolutoinal Layer)
class Conv2Dtranspose(object):
    '''
      constructor's args:
          input      : input image (2D matrix)
          output_siz : output image size
          in_ch      : number of incoming image channel
          out_ch     : number of outgoing image channel
          patch_siz  : filter(patch) size
    '''

    def __init__(self, input, output_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input
        self.rows = output_siz[0]
        self.cols = output_siz[1]
        self.out_ch = out_ch
        self.activation = activation

        wshape = [patch_siz[0], patch_siz[1], out_ch, in_ch]  # note the arguments order

        w_cvt = tf.Variable(tf.truncated_normal(wshape, stddev=0.1),
                            trainable=True)
        b_cvt = tf.Variable(tf.constant(0.1, shape=[out_ch]),
                            trainable=True)
        self.batsiz = tf.shape(input)[0]
        self.w = w_cvt
        self.b = b_cvt
        self.params = [self.w, self.b]

    def output(self):
        shape4D = [self.batsiz, self.rows, self.cols, self.out_ch]
        linout = tf.nn.conv2d_transpose(self.input, self.w, output_shape=shape4D,
                                        strides=[1, 2, 2, 1], padding='SAME') + self.b
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        else:
            self.output = linout

        return self.output


def mk_nn_model(x, y_):
    # Encoding phase
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = Convolution2D(x_image, (28, 28), 1, 16,
                          (3, 3), activation='relu')
    conv1_out = conv1.output()

    pool1 = MaxPooling2D(conv1_out)
    pool1_out = pool1.output()

    conv2 = Convolution2D(pool1_out, (14, 14), 16, 8,
                          (3, 3), activation='relu')
    conv2_out = conv2.output()

    pool2 = MaxPooling2D(conv2_out)
    pool2_out = pool2.output()

    conv3 = Convolution2D(pool2_out, (7, 7), 8, 8, (3, 3), activation='relu')
    conv3_out = conv3.output()

    pool3 = MaxPooling2D(conv3_out)
    pool3_out = pool3.output()
    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    # Decoding phase
    conv_t1 = Conv2Dtranspose(pool3_out, (7, 7), 8, 8,
                              (3, 3), activation='relu')
    conv_t1_out = conv_t1.output()

    conv_t2 = Conv2Dtranspose(conv_t1_out, (14, 14), 8, 8,
                              (3, 3), activation='relu')
    conv_t2_out = conv_t2.output()

    conv_t3 = Conv2Dtranspose(conv_t2_out, (28, 28), 8, 16,
                              (3, 3), activation='relu')
    conv_t3_out = conv_t3.output()

    conv_last = Convolution2D(conv_t3_out, (28, 28), 16, 1, (3, 3),
                              activation='sigmoid')
    decoded = conv_last.output()

    decoded = tf.reshape(decoded, [-1, 784])

    return decoded

def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64):
    nb_classes = 10
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(4264)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    sess = tf.Session()

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "./"
    model_name = "clean_trained_mnist_model"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': model_path,
        'filename': model_name
    }
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([443, 224, 39])

    if clean_train:
        model = make_basic_cnn(nb_filters=nb_filters, nb_classes=nb_classes)
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)

        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate, save=True,
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
        # Variables
        xx = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        p_x = tf.reshape(xx, [-1, 28, 28, 1])
        preds_x = model.get_probs(p_x)
        decoded = mk_nn_model(xx, y_)
        p_decoded = tf.reshape(decoded, [-1, 28, 28, 1])
        mse = tf.losses.mean_squared_error(xx, decoded)
        pred_decoded = model.get_probs(p_decoded)
        pred_loss = -abs(tf.losses.absolute_difference(preds_x, pred_decoded))
        loss = tf.reduce_mean(mse + pred_loss)

        train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

        init = tf.initialize_all_variables()
        with sess as sess:
            print('Training...')
            sess.run(init)
            for i in range(10001):
                batch_xs, batch_ys = mnist.train.next_batch(128)
                train_step.run({xx: batch_xs, y_: batch_ys})
                if i % 1000 == 0:
                    train_loss = loss.eval({xx: batch_xs, y_: batch_ys})
                    print('  step, loss = %6d: %6.3f' % (i, train_loss))

            # generate decoded image with test data
            test_fd = {xx: mnist.test.images, y_: mnist.test.labels}
            decoded_imgs = decoded.eval(test_fd)
            print('loss (test) = ', loss.eval(test_fd))
            adv_x = tf.reshape(decoded_imgs, [-1, 28, 28, 1])
            preds_adv = model.get_probs(adv_x)

            # Evaluate the accuracy of the MNIST model on adversarial examples
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
            print('Test accuracy on adversarial examples: %0.4f\n' % acc)
            report.clean_train_adv_eval = acc

            # Calculate training error
            if testing:
                eval_par = {'batch_size': batch_size}
                acc = model_eval(sess, x, y, preds_adv, X_train,
                                 Y_train, args=eval_par)
                report.train_clean_train_adv_eval = acc

        x_test = mnist.test.images
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # plt.show()
        plt.savefig('mnist_ae2.png')

    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)

if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
