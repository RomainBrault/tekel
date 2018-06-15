"""TensorFlow Kernel Library: tekel."""

from .tf_scoped_cache import scope, clear_all_cached_functions
from .tf_summaries import variable_summaries

from .environment import get_precision

__version__ = '0.0.1rc0'
__authors__ = ['Romain Brault']

__all__ = [  # tf_scoped_cache
           'scope', 'clear_all_cached_functions',
             # tf_summaries
           'variable_summaries',
             # environment
           'get_precision']
