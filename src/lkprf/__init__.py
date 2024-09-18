# Standard library
import os  # noqa
import logging

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

__version__ = "1.1.0dev"
logger = logging.getLogger("lkprf")

from .data import *  # noqa
from .keplerprf import KeplerPRF  # noqa
from .tessprf import TESSPRF  # noqa
