"""Test top-level imports of tekel."""

import tekel


def test_version():
    """Test wether the module has a __version__ set."""
    assert(hasattr(tekel, '__version__'))


def test_authors():
    """Test wether the module has an __authors__ set."""
    assert(hasattr(tekel, '__authors__'))


def test_all():
    """Test wether the module has an __all__ list set."""
    assert(hasattr(tekel, '__all__'))
    assert(isinstance(tekel.__all__, list))
    assert(len(tekel.__all__) >= 0)
