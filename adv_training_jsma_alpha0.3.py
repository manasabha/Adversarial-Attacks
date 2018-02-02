from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import time
import warnings
import math
import os

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level, batch_indices, _ArgsWrapper, create_logger
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax, model_loss
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans_tutorials.tutorial_models import make_basic_cnn

_logger = create_logger("cleverhans.utils.tf")
FLAGS = flags.FLAGS


def effective_train_jsma(train_start=0, train_end=2000, test_start=0,
                        test_end=10000, viz_enabled=False, nb_epochs=6,
                        batch_size=128, nb_classes=10, source_samples=10,
                        learning_rate=0.001):
    
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()
    
    # Set logging level to see debug information
    set_log_level(logging.DEBUG)
    
    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")
    
    model_path = "./"
    model_name = "adv_trained_jsma_model_alpha0.3"
    
    # sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2017, 8, 30])

    # Define input TF placeholder
    x1 = tf.placeholder(tf.float32, shape=(None, 28, 28, 1)) # for clean data
    x2 = tf.placeholder(tf.float32, shape=(None, 28, 28, 1)) # for adv data
    y = tf.placeholder(tf.float32, shape=(None, 10)) # for adv clean targets
    
    # Initialize the model
    model = make_basic_cnn()
    preds = model(x1)
    preds_adv = model(x2)
    
    # Instantiate a SaliencyMapMethod attack object
    # jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}
    
    # Define loss
    loss = 0.3*model_loss(y, preds) + 0.7*model_loss(y, preds_adv)

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = train_step.minimize(loss)
    
    def evaluate_2(adv_examples_last_batch, adv_clean_labels_last_batch):
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x1, y, preds, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x2, y, preds_adv, adv_examples_last_batch,
                              adv_clean_labels_last_batch, args=eval_params)
        print('Test accuracy on last batch of adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy
        
    with sess.as_default():
        tf.global_variables_initializer().run()

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
                # re-instantiate Saliency object with new trained model
                jsma = SaliencyMapMethod(model, back='tf', sess=sess)
                print('--------------------------------------')
                # create an array for storing adv examples
                print('batch: %i/%i' % (batch+1, nb_batches))
                # adv_examples = np.empty([1,28,28,1])
                adv_examples = []
                # for target labels
                #adv_targets = np.empty([1,10])
                # corresponding clean/correct label
                # adv_clean_labels = np.empty([1,10])
                adv_clean_labels = []
                # correspongding clean data
                # adv_clean_examples = np.empty([1,28,28,1])
                adv_clean_examples = []
                
                for sample_ind in xrange(0, batch_size):
                    
                    print('Attacking input %i/%i' % (sample_ind + 1, batch_size))
                    # Compute batch start and end indices
                    start, end = batch_indices(batch, len(X_train), batch_size)
                    X_this_batch = X_train[index_shuf[start:end]]
                    Y_this_batch = Y_train[index_shuf[start:end]]
                    # Perform one training step
                    # feed_dict = {x: X_train[index_shuf[start:end]],y: Y_train[index_shuf[start:end]]}
                    
                    sample = X_this_batch[sample_ind:(sample_ind+1)] # generate from training data
            
                    # We want to find an adversarial example for each possible target class
                    # (i.e. all classes that differ from the label given in the dataset)
                    current_class = int(np.argmax(Y_this_batch[sample_ind])) # generate from training data
                    target_classes = other_classes(nb_classes, current_class)
                    print('Current class is ', current_class)
            
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
                        adv_x = jsma.generate_np(sample, **jsma_params) # get numpy array (1, 28, 28, 1), not Tensor
                        
                        # Check if success was achieved
                        # res = int(model_argmax(sess, x, preds, adv_x) == target)
                        # if succeeds
                        # if res == 1:
                        # append new adv_x to adv_examples array
                        # append sample here, so that the number of times sample is appended mmatches number of adv_ex.
                        # adv_examples = np.append(adv_examples, adv_x, axis=0)
                        adv_examples.append(adv_x)
                        #adv_targets = np.append(adv_targets, one_hot_target, axis=0)
                        # adv_clean_labels = np.append(adv_clean_labels, np.expand_dims(Y_this_batch[sample_ind],axis=0), axis=0) # generate from training data
                        adv_clean_labels.append(Y_this_batch[sample_ind])
                        # adv_clean_examples = np.append(adv_clean_examples, sample, axis=0)
                        adv_clean_examples.append(sample)
                
                # what we have for this batch, batch_size * 9 data
                # adv_examples = adv_examples[1:,:,:,:]
                #adv_targets = adv_targets[1:,:]
                # adv_clean_labels = adv_clean_labels[1:,:]
                # adv_clean_examples = adv_clean_examples[1:,:,:,:]
                adv_examples = np.reshape(adv_examples, (batch_size*(nb_classes-1),28,28,1))
                adv_clean_examples = np.reshape(adv_clean_examples, (batch_size*(nb_classes-1),28,28,1))
                feed_dict = {x1: adv_clean_examples, x2: adv_examples, y: adv_clean_labels}
                train_step.run(feed_dict=feed_dict)
            
            cur = time.time()
            _logger.info("Epoch " + str(epoch) + " took " + str(cur - prev) + " seconds")
            
            evaluate_2(adv_examples, adv_clean_labels)
        print('Training finished.')
        
        # report on clean test data
        preds_test = model(x1)
        eval_par = {'batch_size': 10}
        acc_clean = model_eval(sess, x1, y, preds_test, X_test, Y_test, args=eval_par)
        print('Test accuracy on legitimate examples: %0.4f\n' % acc_clean)
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
        
def main(argv=None):
    effective_train_jsma(viz_enabled=FLAGS.viz_enabled,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        nb_classes=FLAGS.nb_classes,
                        source_samples=FLAGS.source_samples,
                        learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 3, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
        
        
                
                            
                            
                            
                            
                            
                            