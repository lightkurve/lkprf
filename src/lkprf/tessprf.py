"""Class to work with TESS data PRF"""

from typing import Tuple, List
import numpy as np
from .utils import LKPRFWarning
from .data import get_tess_prf_file
import warnings

from .prfmodel import PRF


class TESSPRF(PRF):
    """A TESSPRF class. The TESS PRF measurements are supersampled by a factor of 9."""

    def __init__(self, camera: int, ccd: int):
        super().__init__()
        self.camera = camera
        self.ccd = ccd
        self.mission = "TESS"
        self._prepare_prf()

    def __repr__(self):
        return f"TESSPRF Object [Camera {self.camera}, CCD {self.ccd}]"

    def _get_prf_data(self):
        return get_tess_prf_file(camera=self.camera, ccd=self.ccd)

    def update_coordinates(self, targets: List[Tuple], shape: Tuple):
        row, column = self._unpack_targets(targets)
        """Check that coordinates are within bounds, raise warnings otherwise."""
        if (np.atleast_1d(column) < 45).any():
            warnings.warn(
                "`targets` contains collateral pixels: Column(s) < 45",
                LKPRFWarning,
            )

        if ((np.atleast_1d(column) + shape[1]) > 2093).any():
            warnings.warn(
                "`targets` contains collateral pixels: Column(s) > 2093 ",
                LKPRFWarning,
            )
        if (np.atleast_1d(row) < 0).any():
            warnings.warn(
                "`targets` contains collateral pixels: Row(s) < 0",
                LKPRFWarning,
            )
        if ((np.atleast_1d(row) + shape[0]) > 2048).any():
            warnings.warn(
                "`targets` contains collateral pixels: Row(s) > 2048)",
                LKPRFWarning,
            )
        self._update_coordinates(targets=targets, shape=shape)
        return
