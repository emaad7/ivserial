"""ivserial: exact symbolic tests of serial independence.

    >>> from ivserial import serial_test
    >>> res = serial_test(x, seed=1)
    >>> print(res)
"""
__version__ = "1.0.0"

from .core import (serial_test, SerialTestResult, choose_resolution, null_limit_sf, null_limit_quantile,
                   null_limit_summary, required_sample_size, approximate_power, sample_size_table,
                   serial_cramer_v, gamma_from_cramer_v)
from .io import read_table

__all__ = ["serial_test", "SerialTestResult", "choose_resolution", "null_limit_sf", "null_limit_quantile",
           "null_limit_summary", "required_sample_size", "approximate_power", "sample_size_table",
           "serial_cramer_v", "gamma_from_cramer_v", "read_table", "__version__"]
