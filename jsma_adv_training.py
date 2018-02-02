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

FLAGS = flags.FLAGS

"""
This is one way to do adversarial training using adversarial sample data 
augmentation.
"""
# def reload_data(fn):
#     with np.load(fn) as data:
#         adv_examples, adv_targets, adv_clean_labels = data['adv_examples'], data['adv_targets'], data['adv_clean_labels']
#         return adv_examples, adv_targets, adv_clean_labels
    

def model_run(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=False,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64):
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(7076)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    sess = tf.Session()
    
    rng = np.random.RandomState([2017, 11, 1])
    
    # Get MNIST clean data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    
    # Get adversarial data
    with np.load("adversarial_mnist_test_from_1500.npz") as data:
        adv_X_test, adv_Y_test, adv_clean_Y_test, adv_clean_X_test = data['adv_examples'], data['adv_targets'], data['adv_clean_labels'], data['adv_clean_examples']
    with np.load("adversarial_mnist_train_from_6000.npz") as data:
        adv_X_train, adv_Y_train, adv_clean_Y_train, adv_clean_X_train = data['adv_examples'], data['adv_targets'], data['adv_clean_labels'], data['adv_clean_examples']
    print('Adversarial data are successfully reloaded.')
    
    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds, adv_X_test,
                              adv_clean_Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy
    
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # the second option: we concatenate clean training data and adversarial training data, and do classic/traditional training
    model = make_basic_cnn(nb_filters=nb_filters)
    preds = model.get_probs(x)
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    training_data = np.concatenate((X_train, adv_X_train), axis=0)
    training_target = np.concatenate((Y_train, adv_clean_Y_train), axis=0)
    print("the shape of training data for adversarial training: ", np.shape(training_data))
    
    # test_data = np.concatenate((X_test, adv_X_test), axis=0)
    # test_target = np.concatenate((Y_test, adv_clean_Y_test), axis=0)
    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds, training_data, training_target,
                evaluate=evaluate_2,
                args=train_params, rng=rng)
    
    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds, training_data, training_target,
                              args=eval_params)
        report.train_adv_train_adv_eval = accuracy
    with np.load("adv_test_fgsm_data.npz") as data:
        adv_X_test, adv_clean_Y_test, adv_clean_X_test = data['adv_examples'], data['adv_clean_labels'], data['adv_clean_examples']
    print('Adversarial data are successfully reloaded.')

    preds_adv = model(x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, adv_X_test, adv_clean_Y_test, args=eval_par)
    print('Test accuracy on adversarial examples of fgsm: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    return report


def main(argv=None):
    model_run(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', False, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    
    tf.app.run()
