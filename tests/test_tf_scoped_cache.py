import tensorflow as tf
import numpy as np

from tekel.tf_scoped_cache import *


tf.enable_eager_execution()


def test_tf_scoped_cache():

    tf.executing_eagerly()

    @scope('constant')
    def create_constant(cte):
        return tf.constant(cte)

    c = create_constant(1.)
    assert(create_constant.cache_info().hits == 0)
    assert(create_constant.cache_info().misses == 1)
    d = create_constant(1.)
    assert(create_constant.cache_info().hits == 1)
    assert(create_constant.cache_info().misses == 1)
    assert((c + d).numpy() == 2)

    clear_all_cached_functions()


def test_tf_scoped_cache_diff_type():

    tf.executing_eagerly()

    @scope('constant')
    def create_constant(cte):
        return tf.constant(cte)

    c = create_constant(np.array(1., dtype=np.float32))
    assert(create_constant.cache_info().hits == 0)
    assert(create_constant.cache_info().misses == 1)
    d = create_constant(1.)
    assert(create_constant.cache_info().hits == 0)
    assert(create_constant.cache_info().misses == 2)
    assert((c + d).numpy() == 2)

    clear_all_cached_functions()


def test_tf_scoped_cache_kwards():

    tf.executing_eagerly()

    @scope('constant')
    def create_constant(cte=1):
        return tf.constant(cte)

    c = create_constant(cte=1)
    assert(create_constant.cache_info().hits == 0)
    assert(create_constant.cache_info().misses == 1)
    d = create_constant(cte=1)
    assert(create_constant.cache_info().hits == 1)
    assert(create_constant.cache_info().misses == 1)
    assert((c + d).numpy() == 2)

    clear_all_cached_functions()
