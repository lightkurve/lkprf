# Standard library
import os  # noqa
import logging

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

version = "1.0.0"
logger = logging.getLogger("lkprf")

from .data import *  # noqa
from .keplerprf import KeplerPRF  # noqa
from .tessprf import TESSPRF  # noqa
