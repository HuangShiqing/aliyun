# -*- coding: utf-8 -*-

import tensorflow as tf




## Cost Functions


def cross_entropy(output, target, name=None):
    """It is a softmax cross-entropy operation, returns the TensorFlow expression of cross-entropy of two distributions, implement
    softmax internally. See ``tf.nn.sparse_softmax_cross_entropy_with_logits``.
    Parameters
    ----------
    output : Tensorflow variable
        A distribution with shape: [batch_size, n_feature].
    target : Tensorflow variable
        A batch of index with shape: [batch_size, ].
    name : string
        Name of this loss.
    Examples
    --------
    >>> ce = tl.cost.cross_entropy(y_logits, y_target_logits, 'my_loss')
    References
    -----------
    - About cross-entropy: `wiki <https://en.wikipedia.org/wiki/Cross_entropy>`_.\n
    - The code is borrowed from: `here <https://en.wikipedia.org/wiki/Cross_entropy>`_.
    """
    # try: # old
    #     return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, targets=target))
    # except: # TF 1.0
    assert name is not None, "Please give a unique name to tl.cost.cross_entropy for TF1.0+"
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output, name=name))