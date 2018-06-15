"""Process environement variable."""

import os

import tensorflow as tf

from .utils import default

__all__ = ['get_precision']
__SUPPORTED_PRECISIONS__ = ['fp32', 'fp64']
__DEFAULT_PRECISION__ = __SUPPORTED_PRECISIONS__[0]


tf.app.flags.DEFINE_string('precision',
                           __DEFAULT_PRECISION__,
                           'Define which precision to use for computations. '
                           'should be in {}'.format(__SUPPORTED_PRECISIONS__))

tf.app.flags.DEFINE_string('cov',
                           '',
                           'Dummy flag for compatibility with pytest-cov')


def get_precision():
    """Get the environment precision.

    Returns
    -------
    precision : str
        returns the environment variable 'ITL_PRECISION'. If the environment
        variable doesn't exists return 'fp32'.

    Raises
    ------
    NotImplementedError
        The environment variable 'ITL_PRECISION' is set to something different
        from 'fp32' or 'fp64'.

    """
    precision = default(os.environ.get('TEKEL_PRECISION'),
                        tf.app.flags.FLAGS.precision)

    if precision not in __SUPPORTED_PRECISIONS__:
        raise NotImplementedError('Only Floating point precisions {} are '
                                  'supported. {} is unsupported.'
                                  .format(__SUPPORTED_PRECISIONS__, precision))
    return precision
