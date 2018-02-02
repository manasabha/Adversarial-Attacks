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
import math
import time
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_loss
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level, batch_indices, create_logger

import os

_logger = create_logger("cleverhans.utils.tf")
FLAGS = flags.FLAGS


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

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

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

    model_path = "./"
    model_name = "adv_trained_fgsm_model_mix_data_notmnist"
    
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([1992, 8, 3])

    
    model = make_basic_cnn(nb_filters=nb_filters)
    preds = model(x)
    
    # Create TF session
    sess = tf.Session()
    
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model(adv_x)
    mixed_x = tf.concat([x, adv_x], 0)
    mixed_y = tf.concat([y, y], 0)
    # length = tf.shape(mixed_x)[0]
    index_shuffle = list(range(batch_size*2))
    rng.shuffle(index_shuffle)
    mixed_x = tf.gather(mixed_x, index_shuffle)
    mixed_y = tf.gather(mixed_y, index_shuffle)
    preds_mixed = model(mixed_x)
    
    
    loss = model_loss(mixed_y, preds_mixed)

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = train_step.minimize(loss)

    
    tf.global_variables_initializer().run(session=sess)

    for epoch in xrange(nb_epochs):
        print('Training for epoch %i/%i' % (epoch, nb_epochs-1))
        
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
        assert nb_batches * batch_size >= len(X_train)

        # Indices to shuffle training set
        index_shuf = list(range(len(X_train)))
        rng.shuffle(index_shuf)

        prev = time.time()
        for batch in range(nb_batches):
            # re-instantiate FGSM object with new trained model
            # fgsm = FastGradientMethod(model, sess=sess)
            # adv_x = fgsm.generate(x, **fgsm_params)
            print('--------------------------------------')
            # create an array for storing adv examples
            print('batch: %i/%i' % (batch+1, nb_batches))
            # adv_examples = np.empty([1,28,28,1])
            start, end = batch_indices(batch, len(X_train), batch_size)
            X_this_batch = X_train[index_shuf[start:end]]
            Y_this_batch = Y_train[index_shuf[start:end]]
            
            # adv_examples = sess.run(adv_x, feed_dict={x:X_this_batch})
            # for target labels
            #adv_targets = np.empty([1,10])
            # corresponding clean/correct label
            # adv_clean_labels = np.empty([1,10])
            # correspongding clean data
            # adv_clean_examples = np.empty([1,28,28,1])
            
            
            
            # adv_examples = np.reshape(adv_examples, (batch_size*(nb_classes-1),28,28,1))
            # adv_clean_examples = np.reshape(adv_clean_examples, (batch_size*(nb_classes-1),28,28,1))
            # mixed_X = np.concatenate((X_this_batch, adv_examples), axis=0)
            # mixed_Y = np.concatenate((Y_this_batch, Y_this_batch), axis=0)
            # print('mixed data have shape', np.shape(mixed_X))
            # print('mixed labels have shape', np.shape(mixed_Y))
            
            #shuffle the mixed data before training
            # index_of_batch = list(range(np.shape(mixed_Y)[0]))
            # rng.shuffle(index_of_batch)
            # mixed_X = mixed_X[index_of_batch]
            # mixed_Y = mixed_Y[index_of_batch]
            feed_dict = {x: X_this_batch,  y: Y_this_batch}
            train_step.run(feed_dict=feed_dict, session=sess)
        
        cur = time.time()
        _logger.info("Epoch " + str(epoch) + " took " + str(cur - prev) + " seconds")
        
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)
        
        acc2 = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_params)
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on adversarial examples: %0.4f' % acc2)
        
    print('Training finished.')
    
    # reload fgsm successfully attacking adv test data
    # with np.load("adversarial_fgsm.npz") as data:
    #     adv_X_test, adv_clean_Y_test, adv_clean_X_test = data['adv_examples'], data['adv_clean_labels'], data['adv_clean_examples']
    # print('FGSM adversarial data are successfully reloaded.')
    # preds_adv_test = model(x1)
    # # Evaluate the accuracy of the MNIST model on adversarial examples
    # # eval_par = {'batch_size': 10}
    # acc = model_eval(sess, x1, y, preds_adv_test, adv_X_test, adv_clean_Y_test, args=eval_par)
    # print('Test accuracy on pre-generated adversarial examples of fgsm: %0.4f\n' % acc)
    # # reload fgsm successfully attacking adv test data
    # with np.load("adversarial_mnist_test_from_1500.npz") as data:
    #     adv_X_test, adv_clean_Y_test, adv_clean_X_test = data['adv_examples'], data['adv_clean_labels'], data['adv_clean_examples']
    # print('JSMA adversarial data are successfully reloaded.')
    # # Evaluate the accuracy of the MNIST model on adversarial examples
    # acc2 = model_eval(sess, x1, y, preds_adv_test, adv_X_test, adv_clean_Y_test, args=eval_par)
    # print('Test accuracy on pre-generated adversarial examples of jsma: %0.4f\n' % acc2)
    save_path = os.path.join(model_path, model_name)
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    _logger.info("Completed model training and saved at: " + str(save_path))
    # Close TF session
    sess.close()

    return


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 12, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', False, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
