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
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS
"""
Attack using FGSM generated examples on nonMNIST dataset
"""

def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()
    model_path = "./"
    model_name = "clean_trained__model_notmnist"
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(7895)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    

    # Get MNIST test data
    # X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
    #                                               train_end=train_end,
    #                                               test_start=test_start,
    #                                               test_end=test_end)
    # Get notMNIST data
    with np.load("notmnist.npz") as data:
        X_train, Y_train, X_test, Y_test = data['examples_train'], data['labels_train'], data['examples_test'], data['labels_test']

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    
    # Define TF model graph
    model = make_basic_cnn()
    
    # Create TF session
    sess = tf.Session()
    
    if tf_model_load(sess, file_path=os.path.join(model_path, model_name)):
        print(model_name, " reloaded.")
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.get_probs(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc
    
    
    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 8, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', False, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
