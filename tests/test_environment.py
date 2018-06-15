"""Test tekel environment.py module.:"""

import os
import pytest

import tekel.environment as env


def test_get_precision_float32():
    """Test environment variable TEKEL_PRECISION set to 'fp32'."""
    os.environ['TEKEL_PRECISION'] = 'fp32'
    precision = env.get_precision()
    del os.environ['TEKEL_PRECISION']
    assert(precision == 'fp32')


def test_get_precision_float64():
    """Test environment variable TEKEL_PRECISION set to 'fp64'."""
    os.environ['TEKEL_PRECISION'] = 'fp64'
    precision = env.get_precision()
    del os.environ['TEKEL_PRECISION']
    assert(precision == 'fp64')


def test_get_precision_failure():
    """Test if raise when environment variable TEKEL_PRECISION is set to
       'int32'."""
    os.environ['TEKEL_PRECISION'] = 'int32'
    with pytest.raises(NotImplementedError):
        env.get_precision()
    del os.environ['TEKEL_PRECISION']
