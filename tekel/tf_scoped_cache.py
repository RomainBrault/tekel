"""Scoped TensorFlow function cache."""

import tensorflow as tf
import numpy as np

from collections import namedtuple
from functools import update_wrapper
from threading import RLock

__all__ = ['scope', 'clear_all_cached_functions']

_CACHE_INFO = namedtuple("CacheInfo", ["hits", "misses", "currsize"])
_CACHED_NODES = []


def _make_key(args, kwds,
              kwd_mark=(object(),),
              sorted=sorted, tuple=tuple, type=type, len=len):
    'Make a cache key from a positional and keyword arguments'
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    # numpy trick
    key = ((item.tostring()
            if isinstance(item, np.ndarray)
            else item) for item in key)
    return ''.join(map(str, map(hash, map(str, key))))


def scoped_cache(scope_name):
    """Scoped cache decorator.

    Arguments to the cached function must be hashable or numpy.ndarray.
    View the cache statistics named tuple (hits, misses, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.
    """

    # Users should only access the cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    def decorating_function(user_function):

        cache = dict()
        stats = [0, 0]          # make statistics updateable non-locally
        HITS, MISSES = 0, 1     # names for the stats fields
        make_key = _make_key
        cache_get = cache.get   # bound method to lookup key or return None
        lock = RLock()          # because linkedlist updates aren't threadsafe
        root = []               # root of the circular doubly linked list
        root[:] = [root, root, None, None]  # initialize by pointing to self
        nonlocal_root = [root]                  # make updateable non-locally

        def wrapper(*args, **kwds):
            # simple caching without ordering or size limit
            key = make_key(args, kwds)
            # root used here as a unique not-found sentinel
            result = cache_get(key, root)
            if result is not root:
                stats[HITS] += 1
                return result
            with tf.variable_scope(scope_name):
                result = user_function(*args, **kwds)
            cache[key] = result
            stats[MISSES] += 1
            return result

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CACHE_INFO(stats[HITS], stats[MISSES], len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            with lock:
                cache.clear()
                root = nonlocal_root[0]
                root[:] = [root, root, None, None]
                stats[:] = [0, 0]

        wrapper.__wrapped__ = user_function
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)

    return decorating_function


def scope(scope_name):
    """Scoped cache decorator.

    This function cache a tensorflow node, add a scope with name scope_name and
    add the cached not to a global variable tracking the cached nodes.

    Arguments to the cached nodes must be hashable or numpy.ndarray.
    View the cache statistics named tuple (hits, misses, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    All nodes cached with this function can be uncached using the methods
    clear_all_cached_functions.
    """

    def decorator(func):
        func = scoped_cache(scope_name)(func)
        _CACHED_NODES.append(func)
        return func

    return decorator


def clear_all_cached_functions():
    """ Remove all node cached with the method scope from the cache.
    """
    for node in _CACHED_NODES:
        node.cache_clear()
