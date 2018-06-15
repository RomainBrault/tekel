"""Utility methods for tekel."""

__all__ = ['default']


def default(val, alt):
    """Set a default value to a variable.

    Parameters
    ----------
    val : Any
        The variable to which to set a default value.

    alt : Any
        The default value.

    Returns
    -------
    res : Any
        res is set to alt if val is 'None', otherwise res is set to val.

    """
    if val is None:
        return alt
    else:
        return val
