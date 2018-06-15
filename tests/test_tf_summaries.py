"""Test tf_summaries.py modules."""

import os, pytest

import tensorflow as tf

from tekel.tf_summaries import variable_summaries
from tekel.environment import __SUPPORTED_PRECISIONS__

try:
    tf.enable_eager_execution()
except ValueError:
    pass


@pytest.mark.parametrize('precision', __SUPPORTED_PRECISIONS__)
def test_variable_summaries_fp32(precision):
    """Test if adding a summary doesn't change a TensorFlow variable."""

    tf.executing_eagerly()

    os.environ['TEKEL_PRECISION'] = precision
    cte = tf.constant(1)
    variable_summaries(cte)
    assert(cte.numpy() == 1)
