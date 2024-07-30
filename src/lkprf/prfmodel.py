"""PRF base class"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy.typing as npt
import numpy as np
from scipy.interpolate import RectBivariateSpline


class PRF(ABC):
    @abstractmethod
    def __init__(self):
        """
        A generic base class object for PRFs. No to be used directly.
        See KeplerPRF and TESSPRF for the instantiable classes.

        Parameters:
        -----------
        column : int
                pixel coordinate of the lower left column value
        row : int
                pixel coordinate of the lower left row value
        shape : tuple
                shape of the resultant PRFs in pixels
        """

    def __repr__(self):
        return "PRF Base Class"

    def __call__(
        self,
        targets: List[Tuple] = [(5.5, 5.5)],
        shape: Tuple = (11, 11),
        origin: Tuple = (0, 0),
    ) -> npt.ArrayLike:
        return self.evaluate(targets=targets, shape=shape, origin=origin)

    def _unpack_targets(self, targets):
        """Take input targets and convert them to row/column arrays"""
        if isinstance(targets, tuple):
            targets = [targets]
        if not isinstance(targets, list):
            raise ValueError("Input a list of targets.")
        if not isinstance(targets[0], tuple):
            raise ValueError(
                "Input targets as tuples with format (row position, column position)."
            )

        target_row, target_column = np.asarray(targets).T
        return target_row, target_column

    def _evaluate(
        self,
        targets: List[Tuple] = [(5.5, 5.5)],
        origin: Tuple = (0, 0),
        shape: Tuple = (11, 11),
        dx=0,
        dy=0,
    ):
        """
        Interpolates the PRF model onto detector coordinates. Hidden function

        Parameters
        ----------
        targets : List of Tuples
            Coordinates of the targets
        origin : Tuple
            The origin of the image, combined with shape this sets the extent of the image
        shape : Tuple
            The shape of the image, combined with the origin this sets the extent of the image

        Returns
        -------
        prf : 3D array
            Three dimensional array representing the PRF values parametrized by flux and centroids.
            Has shape (ntargets, shape[0], shape[1])
        """
        self.update_coordinates(targets=targets, shape=shape)
        target_row, target_column = self._unpack_targets(targets)

        # Integer extent from the PRF model
        r1, r2 = int(np.floor(self.PRFrow[0])), int(np.ceil(self.PRFrow[-1]))
        c1, c2 = int(np.floor(self.PRFcol[0])), int(np.ceil(self.PRFcol[-1]))
        # Position in the PRF model for each source position % 1
        delta_row, delta_col = (
            np.arange(r1, r2)[:, None] + 0.5 - np.atleast_1d(target_row) % 1,
            np.arange(c1, c2)[:, None] + 0.5 - np.atleast_1d(target_column) % 1,
        )

        # prf model for each source, downsampled to pixel grid
        prf = np.asarray(
            [
                self.interpolate(dr, dc, dx=dx, dy=dy)
                for dr, dc in zip(delta_row.T, delta_col.T)
            ]
        )

        prf[np.abs(prf) < 1e-6] = 0

        # Normalize to ensure no flux loss
        if (dx == 0) & (dy == 0):
            # PRF model should sum to one
            prf /= prf.sum(axis=(1, 2))[:, None, None]

        # Insert values into final array of the correct shape
        ar = np.zeros((len(target_column), *shape))
        for idx, r, c, p in zip(range(len(prf)), target_row, target_column, prf):
            # pixel offset for source
            roffset = int(r - r % 1)
            coffset = int(c - c % 1)
            # row and column position for source
            R, C = np.arange(r1 + roffset, r2 + roffset), np.arange(
                c1 + coffset, c2 + coffset
            )
            # check if pixels are in the resultant image
            k = (R >= origin[0]) & (R < (origin[0] + shape[0]))
            j = (C >= origin[1]) & (C < (origin[1] + shape[1]))
            if k.any() & j.any():
                # if yes, insert into the resultant image
                X, Y = np.meshgrid(R[k] - origin[0], C[j] - origin[1], indexing="ij")
                ar[idx, X, Y] = p[k][:, j]
        return ar

    def evaluate(
        self,
        targets: List[Tuple] = [(5.5, 5.5)],
        origin: Tuple = (0, 0),
        shape: Tuple = (11, 11),
    ):
        """
        Interpolates the PRF model onto detector coordinates.

        Parameters
        ----------
        targets : List of Tuples
            Coordinates of the targets
        origin : Tuple
            The origin of the image, combined with shape this sets the extent of the image
        shape : Tuple
            The shape of the image, combined with the origin this sets the extent of the image

        Returns
        -------
        prf : 3D array
            Three dimensional array representing the PRF values parametrized by flux and centroids.
            Has shape (ntargets, shape[0], shape[1])
        """
        self.update_coordinates(targets=targets, shape=shape)
        return self._evaluate(targets=targets, shape=shape, origin=origin, dx=0, dy=0)

    def gradient(
        self,
        targets: List[Tuple] = [(5.5, 5.5)],
        origin: Tuple = (0, 0),
        shape: Tuple = (11, 11),
    ) -> Tuple:
        """
        Interpolates the gradient of the PRF model onto detector coordinates.

        Parameters
        ----------
        targets : List of Tuples
            Coordinates of the targets
        origin : Tuple
            The origin of the image, combined with shape this sets the extent of the image
        shape : Tuple
            The shape of the image, combined with the origin this sets the extent of the image

        Returns
        -------
        deriv_row, deriv_col : Tuple of two 3D arrays
            This tuple contains two 3D arrays representing the gradient of the PRF values parametrized by flux and centroids.
            Returns (gradient in row, gradient in column). Each array has shape (ntargets, shape[0], shape[1])
        """
        self.update_coordinates(targets=targets, shape=shape)
        deriv_col = self._evaluate(
            targets=targets, shape=shape, origin=origin, dx=0, dy=1
        )
        deriv_row = self._evaluate(
            targets=targets, shape=shape, origin=origin, dx=1, dy=0
        )
        return deriv_row, deriv_col

    @abstractmethod
    def _get_prf_data(self):
        """Method to open PRF files for given mission"""
        pass

    def _prepare_prf(self):
        """
        Sets up the PRF model interpolation by reading in the relevant files,
        and combining them by weighting them by distance to the location on the CCD of interest
        """

        hdulist = self._get_prf_data()
        PRFdata, crval1p, crval2p, cdelt1p, cdelt2p = [], [], [], [], []
        for hdu in hdulist[1:]:
            PRFdata.append(hdu.read())
            hdr = hdu.read_header()
            crval1p.append(hdr["CRVAL1P"])
            crval2p.append(hdr["CRVAL2P"])
            cdelt1p.append(hdr["CDELT1P"])
            cdelt2p.append(hdr["CDELT2P"])

        PRFdata, crval1p, crval2p, cdelt1p, cdelt2p = (
            np.asarray(PRFdata),
            np.asarray(crval1p),
            np.asarray(crval2p),
            np.asarray(cdelt1p),
            np.asarray(cdelt2p),
        )
        PRFdata /= PRFdata.sum(axis=(1, 2))[:, None, None]

        PRFcol = np.arange(0, np.shape(PRFdata[0])[1] + 0)
        PRFrow = np.arange(0, np.shape(PRFdata[0])[0] + 0)

        # Shifts pixels so it is in pixel units centered on 0
        PRFcol = (PRFcol - np.size(PRFcol) / 2) * cdelt1p[0]
        PRFrow = (PRFrow - np.size(PRFrow) / 2) * cdelt2p[0]

        PRFcol += 0.5
        PRFrow += 0.5

        (
            self.PRFrow,
            self.PRFcol,
            self.PRFdata,
            self.crval1p,
            self.crval2p,
            self.cdelt1p,
            self.cdelt2p,
        ) = (PRFrow, PRFcol, PRFdata, crval1p, crval2p, cdelt1p, cdelt2p)

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

        for i in range(self.PRFdata.shape[0]):
            prf_weight = np.sqrt(
                (ref_column - self.crval1p[i]) ** 2 + (ref_row - self.crval2p[i]) ** 2
            )
            if prf_weight < min_prf_weight:
                prf_weight = min_prf_weight
            supersamp_prf += self.PRFdata[i] / prf_weight

        supersamp_prf /= np.nansum(supersamp_prf) * self.cdelt1p[0] * self.cdelt2p[0]
        self.interpolate = RectBivariateSpline(self.PRFrow, self.PRFcol, supersamp_prf)
        return

    @abstractmethod
    def update_coordinates(self, targets, shape):
        """Method to update the interpolation function

        Wrap this parent method, use the public method to check that e.g. targets are in bounds.
        This method should provide the interpolation function
        """
        pass
