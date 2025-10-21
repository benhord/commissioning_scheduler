__version__ = "0.1.0"
# Standard library
import logging  # noqa: E402
import os  # noqa
import time  # noqa: E402
from glob import glob
from threading import Event, Thread  # noqa: E402

# Standard library
import configparser  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa

# Third-party
import numpy as np  # noqa: E402


def get_version():
    try:
        return version("packagename")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))


from .scheduler import *  # noqa: E402
