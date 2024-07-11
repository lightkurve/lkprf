"""Class to work with Kepler data PRF"""

from typing import Tuple, List
import numpy as np
from .utils import channel_to_module_output, LKPRFWarning
from .data import get_kepler_prf_file
import warnings

from .prfmodel import PRF


class KeplerPRF(PRF):
    """
    A KeplerPRF class. Can be used for Kepler or K2.

    There are 5 PRF measurements (the 4 corners and the center) for each channel.
    The measured PRF is over-sampled by a factor of 50 to enable for sub-pixel interpolation.
    The model is a 550x550 (or 750x750) grid that covers 11x11 (or 15x15) pixels

    https://archive.stsci.edu/missions/kepler/commissioning_prfs/
    """

    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel
        self.mission = "Kepler"
        self._prepare_prf()

    def __repr__(self):
        return f"KeplerPRF Object [Channel {self.channel}]"

    def _get_prf_data(self):
        module, output = channel_to_module_output(self.channel)
        return get_kepler_prf_file(module=module, output=output)

    def update_coordinates(self, targets: List[Tuple], shape: Tuple):
        row, column = self._unpack_targets(targets)
        """Check that coordinates are within bounds, raise warnings otherwise."""
        if (np.atleast_1d(column) < 12).any():
            warnings.warn(
                "`targets` contains collateral pixels: Column(s) < 12",
                LKPRFWarning,
            )

        if ((np.atleast_1d(column) + shape[1]) > 1112).any():
            warnings.warn(
                "`targets` contains collateral pixels: Column(s) > 1112 ",
                LKPRFWarning,
            )
        if (np.atleast_1d(row) < 20).any():
            warnings.warn(
                "`targets` contains collateral pixels: Row(s) < 20",
                LKPRFWarning,
            )
        if ((np.atleast_1d(row) + shape[0]) > 1044).any():
            warnings.warn(
                "`targets` contains collateral pixels: Row(s) > 1044)",
                LKPRFWarning,
            )
        self._update_coordinates(targets=targets, shape=shape)
        return
