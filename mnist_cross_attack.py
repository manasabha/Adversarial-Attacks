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
# add dependencies
from distutils.version import LooseVersion
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans_tutorials.tutorial_models import make_basic_cnn


import os

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

    # Create TF session
    sess = tf.Session()

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = make_basic_cnn(nb_filters=nb_filters)
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
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        print(adv_x)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        # Define accuracy symbolically
        if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
            correct_preds = tf.not_equal(tf.argmax(y, axis=-1),
                                     tf.argmax(preds_adv, axis=-1))
        else:
            correct_preds = tf.not_equal(tf.argmax(y, axis=tf.rank(y) - 1),
                                     tf.argmax(preds_adv,
                                               axis=tf.rank(preds_adv) - 1))
        # print("the shape of correct_preds is ", correct_preds.get_shape())
        # correct_preds is a boolean Tensor with shape (size,)
        success_adv_x = tf.boolean_mask(adv_x, correct_preds)
        success_clean_x = tf.boolean_mask(x, correct_preds)
        success_clean_y = tf.boolean_mask(y, correct_preds)
        fgsm_adv_x, fgsm_clean_x, fgsm_clean_y = sess.run([success_adv_x, success_clean_x, success_clean_y], feed_dict={x:X_test,y:Y_test})
        np.savez('adversarial_fgsm',adv_examples=fgsm_adv_x, adv_clean_labels=fgsm_clean_y, adv_clean_examples=fgsm_clean_x)
        print("the shape of adversarial examples we save is ", np.shape(fgsm_adv_x))
        print("the shape of clean targets we save is ", np.shape(fgsm_clean_y))

        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples fgsm: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc
        adv_x_test_for_save = sess.run(adv_x, {x: X_test})
        np.savez("adv_test_fgsm_data.npz", adv_examples=adv_x_test_for_save, adv_clean_labels=Y_test, adv_clean_examples=X_test)
        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc

        print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = make_basic_cnn(nb_filters=nb_filters)
    preds_2 = model_2(x)
    fgsm2 = FastGradientMethod(model_2, sess=sess)
    adv_x_2 = fgsm2.generate(x, **fgsm_params)
    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x_2 = tf.stop_gradient(adv_x_2)
    preds_2_adv = model_2(adv_x_2)
    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    # Perform and evaluate adversarial training
    print("pred_adv", preds_2_adv.get_shape())
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng)

    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph


    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    source_samples = 10000
    nb_classes = 10
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((nb_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((nb_classes, source_samples), dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model_2, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    figure = None
    
    # create an array for storing adv examples
    adv_examples = np.empty([1,28,28,1])
    # for target labels
    adv_targets = np.empty([1,10])
    # corresponding clean/correct label
    adv_clean_labels = np.empty([1,10])
    # correspongding clean data
    adv_clean_examples = np.empty([1,28,28,1])
        
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, source_samples):
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
        sample = X_test[sample_ind:(sample_ind+1)] # generate from testing data

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(Y_test[sample_ind])) # generate from testing data
        target_classes = other_classes(nb_classes, current_class)

        # For the grid visualization, keep original images along the diagonal
        # grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
        #     sample, (img_rows, img_cols, channels))
        
        # Loop over all target classes
        for target in target_classes:
            print('Generating adv. example for target class %i' % target)

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            #create fake target
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adv_x = jsma.generate_np(sample, **jsma_params)
            # print('adv_x\'shape is ', np.shape(adv_x)) # (1,28,28,1)
            
            # Check if success was achieved
            res = int(model_argmax(sess, x, preds, adv_x) == target)
            # if succeeds
            if res == 1:
                # append new adv_x to adv_examples array
                # append sample here, so that the number of times sample is appended mmatches number of adv_ex.
                adv_examples = np.append(adv_examples, adv_x, axis=0)
                adv_targets = np.append(adv_targets, one_hot_target, axis=0)
                adv_clean_labels = np.append(adv_clean_labels, np.expand_dims(Y_test[sample_ind],axis=0), axis=0) # generate from testing data
                adv_clean_examples = np.append(adv_clean_examples, sample, axis=0)

            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = X_test[sample_ind].reshape(-1)
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

            # Update the arrays for later analysis
            results[target, sample_ind] = res
            perturbations[target, sample_ind] = percent_perturb
    print('--------------------------------------')
    adv_examples = adv_examples[1:,:,:,:]
    adv_targets = adv_targets[1:,:]
    adv_clean_labels = adv_clean_labels[1:,:]
    adv_clean_examples = adv_clean_examples[1:,:,:,:]
    np.savez('adversarial_jsma_actual_full',adv_examples=adv_examples, adv_targets=adv_targets, adv_clean_labels=adv_clean_labels,adv_clean_examples=adv_clean_examples)
    print(np.shape(adv_targets)[0], "adversarial examples have been saved.")
    
    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    nb_targets_tried = ((nb_classes - 1) * source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
    report.clean_test_adv_eval = 1. - succ_rate

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.4f}'.format(percent_perturb_succ))


    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()

