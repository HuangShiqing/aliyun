# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import os

import tl_layer
import input_data


def inference(images, batch_size):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
        batch_size: batch_size
    Returns:
        y: output tensor with the computed logits, float, [batch_size, n_classes]
        network: network
    '''
    network = tl_layer.InputLayer(images, name='input')
    network = tl_layer.Conv2dLayer(network,
                                   act=tf.nn.relu,
                                   shape=[5, 5, 3, 32],  # 32 features for each 5x5 patch
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   name='cnn_layer1')  # output: (?, 208, 208, 32)
    network = tl_layer.PoolLayer(network,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool=tf.nn.max_pool,
                                 name='pool_layer1', )  # output: (?, 104, 104, 32)
    network = tl_layer.Conv2dLayer(network,
                                   act=tf.nn.relu,
                                   shape=[5, 5, 32, 64],  # 64 features for each 5x5 patch
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   name='cnn_layer2')  # output: (?, 104, 104, 64)
    network = tl_layer.PoolLayer(network,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool=tf.nn.max_pool,
                                 name='pool_layer2', )  # output: (?, 52, 52, 64)
    network = tl_layer.Conv2dLayer(network,
                                   act=tf.nn.relu,
                                   shape=[5, 5, 64, 128],  # 32 features for each 5x5 patch
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   name='cnn_layer3')  # output: (?, 52, 52, 128)
    network = tl_layer.PoolLayer(network,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool=tf.nn.max_pool,
                                 name='pool_layer3', )  # output: (?, 26, 26, 128)
    network = tl_layer.Conv2dLayer(network,
                                   act=tf.nn.relu,
                                   shape=[5, 5, 128, 256],  # 64 features for each 5x5 patch
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   name='cnn_layer4')  # output: (?, 26, 26, 256)
    network = tl_layer.PoolLayer(network,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool=tf.nn.max_pool,
                                 name='pool_layer4', )  # output: (?, 13, 13, 256)

    network = tl_layer.FlattenLayer(network, name='flatten_layer')
    # output: (?, 3136)
    network = tl_layer.DropoutLayer(network, keep=0.5, is_fix=True, name='drop1')
    # output: (?, 3136)
    network = tl_layer.DenseLayer(network, n_units=256, act=tf.nn.relu, name='relu1')
    # output: (?, 256)
    network = tl_layer.DropoutLayer(network, keep=0.5, is_fix=True, name='drop2')
    # output: (?, 256)
    network = tl_layer.DenseLayer(network, n_units=2,
                                  act=tf.identity, name='output_layer')
    y = network.outputs
    return y, network


def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32),
                                      labels) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)

    return accuracy
