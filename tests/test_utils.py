"""Test tekel utils.py module.:"""

import tekel.utils as utils


def test_default_1():
    """Test that if a value is set then default return the specified value."""
    val = 1
    assert(utils.default(val, None) == val)


def test_default_None():
    """Test that if a value is not set (None) then default return the
       specified value."""
    val = None
    assert(utils.default(val, 1) is not None)
