# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Adding hyperparameter optimization to:

A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros

"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import rfho as rf

FLAGS = None


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
      Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.
      Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). keep_prob is a scalar placeholder for the probability of
        dropout.
      """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # TODO # Dropout - controls the complexity of the model, prevents co-adaptation of
    # # features.
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # RFHO: We use L2 norm weight penalty instead of dropout at the last layer.

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv, W_fc1, W_fc2


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    """
    Modified MNIST for expert (CNN part) tensorflow tutorial experiment to include real time
    hyperparameter optimization. Hyperparameters being optimized are learning rate for
    ADAM optimizer and coefficient of L2 norm of fully connected part of the network.
    Note that this codes requires ~ 3x (gpu) memory and ~ 4x time compared to the original one
    but should yield a final test error of around 99.4 %

    :param _:
    :return:
    """
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, W_fc1, W_fc2 = deepnn(x)

    # RFHO: collect model variables and "vectorize the model"
    model_vairables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # use adam optimizer:
    optimizer = rf.AdamOptimizer
    w, y_conv, W_fc1, W_fc2 = rf.vectorize_model(model_vairables, y_conv, W_fc1, W_fc2,
                                                 augment=optimizer.get_augmentation_multiplier(),
                                                 suppress_err_out=False)
    # w is now a vector that contains all the weights, y_conv and W_fc2 are the same tensor as earlier,
    # but in the new graph

    # RFHO use cross entropy defined in the package since tensorflow one does not have Hessian,
    # eps is the clipping threshold for cross entropy.
    cross_entropy = tf.reduce_mean(
        rf.cross_entropy_loss(labels=y_, logits=y_conv, eps=1.e-4))
    # RFHO add an L2 regularizer on the last weight matrix, whose weight will be optimized
    rho = tf.Variable(0., name='rho')
    constraints = [rf.positivity(rho)]  # rho >= 0
    iterations_per_epoch = 1100  # with mini batch size of 50
    training_error = cross_entropy + 1/iterations_per_epoch*tf.multiply(
        rho, tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # RFHO define learning rate as an hyperparameter and create the parameter optimization dynamics
    alpha = tf.Variable(1.e-4, name='alpha')
    constraints.append(rf.positivity(alpha))
    dynamics = optimizer.create(w, lr=alpha, loss=training_error)

    # RFHO we want to optimize learning rate and L2 coefficient w.r.t. cross entropy loss on validation set
    hyper_dict = {cross_entropy: [alpha, rho]}
    # RFHO define the hyperparameter optimizer, we use Forward-HG method to compute hyper-gradients and RTHO algorithm
    hyper_opt = rf.HyperOptimizer(dynamics, hyper_dict, rf.ForwardHG, lr=1.e-5)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # RFHO last thing before running: define the example supplier:
    def _train_fd():
        batch = mnist.train.next_batch(50)  # batch size of 50
        return {x: batch[0], y_: batch[1]}

    def _validation_fd():
        return {x: mnist.validation.images, y_: mnist.validation.labels}

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default() as ss:  # RFHO use default session.
        hyper_opt.initialize()  # RFHO this will initialize all the variables, including hyperparameters
        for i in range(200):  # RFHO we run for 200 hyper-iterations
            hyper_opt.run(100, train_feed_dict_supplier=_train_fd,
                          val_feed_dict_suppliers={cross_entropy: _validation_fd},
                          hyper_constraints_ops=constraints)

            # if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=_train_fd())
            val_accuracy, val_error = ss.run([accuracy, cross_entropy], feed_dict=_validation_fd())

            print('step %d, training accuracy %.2f; validation accuracy: %.4f, validation error: %.5f; '
                  'alpha: %.6f, %.5f, rho: %.6f, %.5f'
                  % (i*100, train_accuracy, val_accuracy, val_error, alpha.eval(),
                     hyper_opt.hyper_gradients.hyper_gradients_dict[alpha].eval(),
                     rho.eval(), hyper_opt.hyper_gradients.hyper_gradients_dict[rho].eval()))
            # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        test_accuracy = accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels})
        print('test accuracy %g' % test_accuracy)
        return test_accuracy


def experiment(mnist, optimizer=rf.AdamOptimizer, optimizer_kwargs=None,
               hyper_batch_size=100, T=200, hyper_learning_rate=1.e-4, use_mse=False):
    """
    Modified MNIST for expert (CNN part) tensorflow tutorial experiment to include real time
    hyperparameter optimization. Hyperparameters being optimized are learning rate for
    ADAM optimizer and coefficient of L2 norm of fully connected part of the network.
    Note that this codes requires ~ 3x (gpu) memory and ~ 4x time compared to the original one
    but should yield a final test error of around 99.4 %

    :return:
    """
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784], name='x')

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='y')

    # Build the graph for the deep net
    y_conv, W_fc1, W_fc2 = deepnn(x)

    # RFHO: collect model variables and "vectorize the model"
    model_vairables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # use adam optimizer:
    w, y_conv, W_fc1, W_fc2 = rf.vectorize_model(model_vairables, y_conv, W_fc1, W_fc2,
                                                 augment=optimizer.get_augmentation_multiplier())
    # w is now a vector that contains all the weights, y_conv and W_fc2 are the same tensor as earlier,
    # but in the new graph

    # RFHO use cross entropy defined in the package since tensorflow one does not have Hessian,
    # eps is the clipping threshold for cross entropy.
    if use_mse:
        error = tf.reduce_mean(tf.squared_difference(y_, y_conv), name='error')
    else:
        error = tf.reduce_mean(
            rf.cross_entropy_loss(labels=y_, logits=y_conv, eps=1.e-4), name='error')
    # RFHO add an L2 regularizer on the last weight matrix, whose weight will be optimized
    rho = tf.Variable(0., name='rho')
    constraints = [rf.positivity(rho)]  # rho >= 0
    iterations_per_epoch = 1100  # with mini batch size of 50
    training_error = error + 1/iterations_per_epoch*tf.multiply(
        rho, tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # RFHO define learning rate as an hyperparameter and create the parameter optimization dynamics
    if optimizer_kwargs is None:
        optimizer_kwargs = {'lr': tf.Variable(1.e-4, name='alpha')}
    dynamics = optimizer.create(w, loss=training_error, **optimizer_kwargs)
    constraints += dynamics.get_natural_hyperparameter_constraints()  # add 'usual' constraints for
    # if optimizer is rf.AdamOptimizer:
    #     constraints.append(dynamics.learning_rate.assign(tf.minimum(1.e-3, dynamics.learning_rate)))
    # algorithmic hyperparameters

    # RFHO we want to optimize learning rate and L2 coefficient w.r.t. cross entropy loss on validation set
    hyper_dict = {error: [rho] + dynamics.get_optimization_hyperparameters(only_variables=True)}
    # RFHO define the hyperparameter optimizer, we use Forward-HG method to compute hyper-gradients and RTHO algorithm
    hyper_opt = rf.HyperOptimizer(dynamics, hyper_dict, rf.ForwardHG, lr=hyper_learning_rate)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # RFHO last thing before running: define the example supplier:
    _train_fd = rf.ExampleVisiting(mnist.train, batch_size=50).create_feed_dict_supplier(x, y_)
    _validation_fd = mnist.validation.create_supplier(x, y_)

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():  # RFHO use default session.
        hyper_opt.initialize()  # RFHO this will initialize all the variables, including hyperparameters
        for i in range(T):  # RFHO we run for 200 hyper-iterations
            hyper_opt.run(hyper_batch_size, train_feed_dict_supplier=_train_fd,
                          val_feed_dict_suppliers={error: _validation_fd},
                          hyper_constraints_ops=constraints)

        test_accuracy = accuracy.eval(feed_dict=mnist.test.create_supplier(x, y_)())
        print('test accuracy %g' % test_accuracy)
        return test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
