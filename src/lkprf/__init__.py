# Standard library
import os  # noqa
import logging
from .version import __version__

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger("lkprf")

from .data import *  # noqa
from .keplerprf import KeplerPRF  # noqa
from .tessprf import TESSPRF  # noqa
