"""Class to work with Kepler data PRF"""

from typing import Tuple, List
import numpy as np
from .utils import channel_to_module_output, LKPRFWarning
from .data import get_kepler_prf_file
from . import PACKAGEDIR
import warnings
from scipy.interpolate import RectBivariateSpline

from .prfmodel import PRF
from scipy.interpolate import RectBivariateSpline


class KeplerPRF(PRF):
    """
    A KeplerPRF class. Can be used for Kepler or K2.

    There are 5 PRF measurements (the 4 corners and the center) for each channel.
    The measured PRF is over-sampled by a factor of 50 to enable for sub-pixel interpolation.
    The model is a 550x550 (or 750x750) grid that covers 11x11 (or 15x15) pixels

    https://archive.stsci.edu/missions/kepler/commissioning_prfs/
    """

    def __init__(self, channel: int, cache_dir: str = PACKAGEDIR + "/data/"):
        super().__init__()
        self.channel = channel
        self.mission = "Kepler"
        self.cache_dir = cache_dir
        self._prepare_prf()

    def __repr__(self):
        return f"KeplerPRF Object [Channel {self.channel}]"

    def _get_prf_data(self):
        module, output = channel_to_module_output(self.channel)
        return get_kepler_prf_file(
            module=module, output=output, cache_dir=self.cache_dir
        )

    def check_coordinates(self, targets: List[Tuple], shape: Tuple):
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
        return

    def _prepare_supersamp_prf(self, targets: List[Tuple], shape: Tuple):
        row, column = self._unpack_targets(targets)
        # Set the row and column for the model

        # Create one supersampled PRF for each image, used for all targets in the image
        row, column = np.atleast_1d(row), np.atleast_1d(column)
        row = np.min([np.max([row, row**0 * 20], axis=0), row**0 * 1044], axis=0).mean()
        column = np.min(
            [np.max([column, column**0 * 12], axis=0), column**0 * 1112], axis=0
        ).mean()

        # interpolate the calibrated PRF shape to the target position
        min_prf_weight = 1e-6
        rowdim, coldim = shape[0], shape[1]
        ref_column = column + 0.5 * coldim
        ref_row = row + 0.5 * rowdim
        supersamp_prf = np.zeros(self.PRFdata.shape[1:], dtype="float32")

        # Find the 3 measurements nearest the desired locations
        # Kepler has 5 measurements, so nearest 3 triangulates the measurement
        prf_weights = [
            np.sqrt(
                (ref_column - self.crval1p[i]) ** 2 + (ref_row - self.crval2p[i]) ** 2
            )
            for i in range(self.PRFdata.shape[0])
        ]
        idx = np.argpartition(prf_weights, 3)[:3]

        for i in idx:
            if prf_weights[i] < min_prf_weight:
                prf_weights[i] = min_prf_weight
            supersamp_prf += self.PRFdata[i] / prf_weights[i]

        supersamp_prf /= np.nansum(supersamp_prf) * self.cdelt1p[0] * self.cdelt2p[0]

        # Set up the interpolation function
        self.interpolate = RectBivariateSpline(self.PRFrow, self.PRFcol, supersamp_prf)
        return

    def _update_coordinates(self, targets: List[Tuple], shape: Tuple):
        row, column = self._unpack_targets(targets)
        # Set the row and column for the model
        row, column = np.atleast_1d(row), np.atleast_1d(column)
        row = np.min([np.max([row, row**0 * 20], axis=0), row**0 * 1044], axis=0).mean()
        column = np.min(
            [np.max([column, column**0 * 12], axis=0), column**0 * 1112], axis=0
        ).mean()

        # interpolate the calibrated PRF shape to the target position
        min_prf_weight = 1e-6
        rowdim, coldim = shape[0], shape[1]
        ref_column = column + 0.5 * coldim
        ref_row = row + 0.5 * rowdim
        supersamp_prf = np.zeros(self.PRFdata.shape[1:], dtype="float32")

        # Find the 3 measurements nearest the desired locations
        # Kepler has 5 measurements, so nearest 3 triangulates the measurement
        prf_weights = [
            np.sqrt(
                (ref_column - self.crval1p[i]) ** 2 + (ref_row - self.crval2p[i]) ** 2
            )
            for i in range(self.PRFdata.shape[0])
        ]
        idx = np.argpartition(prf_weights, 3)[:3]

        for i in idx:
            if prf_weights[i] < min_prf_weight:
                prf_weights[i] = min_prf_weight
            supersamp_prf += self.PRFdata[i] / prf_weights[i]

        supersamp_prf /= np.nansum(supersamp_prf) * self.cdelt1p[0] * self.cdelt2p[0]
        self.interpolate = RectBivariateSpline(self.PRFrow, self.PRFcol, supersamp_prf)
        return
