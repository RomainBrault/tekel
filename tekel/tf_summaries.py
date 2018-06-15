"""TensorFlow summaries."""

import tensorflow as tf

from .environment import get_precision

__all__ = ['variable_summaries']


def variable_summaries(var, name=None,
                       mean=True,
                       stddev=True,
                       bounds=True,
                       histogram=True):
    """Attach a lot of summaries to a Tensor.

    Parameters
    ----------
    name : str
        The name of the summary scope.

    var : tf.Tensor
        The tensor to which to attach the summary.

    Returns
    -------
    var : tf.Tensor
        The (unmodified) tensor passed in the arguments.
    """
    precision = get_precision()
    if precision == 'fp32':
        var = tf.cast(var, tf.float32)
    elif precision == 'fp64':
        var = tf.cast(var, tf.float64)
    with tf.name_scope(name):
        if mean:
            with tf.name_scope('mean'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
        if stddev:
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
        if bounds:
            with tf.name_scope('bounds'):
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
        if histogram:
            with tf.name_scope('histogram'):
                tf.summary.histogram('histogram', var)
    return var
