"""PRF base class"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional
import numpy.typing as npt
import numpy as np
from .aperture import Aperture


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
            Pixel coordinates of the target(s).
        origin : Tuple
            The (row, column) origin of the image in pixels.
            Combined with shape this sets the extent of the image.
        shape : Tuple
            The (N_row, N_col) shape of the image.
            Combined with the origin this sets the extent of the image.

        Returns
        -------
        prf : 3D array
            Three dimensional array representing the PRF values parametrized by flux and centroids.
            Has shape (ntargets, shape[0], shape[1])
        """

        # self.update_coordinates(targets=targets, shape=shape)
        target_row, target_column = self._unpack_targets(targets)

        # Integer extent from the PRF model
        r1, r2 = int(np.floor(self.PRFrow[0])), int(np.ceil(self.PRFrow[-1]))
        c1, c2 = int(np.floor(self.PRFcol[0])), int(np.ceil(self.PRFcol[-1]))
        # Position in the PRF model for each source position % 1
        delta_row, delta_col = (
            np.arange(r1, r2)[:, None] - np.atleast_1d(target_row) % 1,
            np.arange(c1, c2)[:, None] - np.atleast_1d(target_column) % 1,
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
            R, C = (
                np.arange(r1 + roffset, r2 + roffset),
                np.arange(c1 + coffset, c2 + coffset),
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
        self.check_coordinates(targets=targets, shape=shape)
        self._prepare_supersamp_prf(targets=targets, shape=shape)
        # Need to define model_prf as a proprety of self here so it can be used in get_aperture
        self.model_prf = self._evaluate(
            targets=targets, shape=shape, origin=origin, dx=0, dy=0
        )
        return self.model_prf

    def get_aperture(
        self,
        aperture_type: str = "snr",
        completeness: float = 0.9,
        crowding_metric: float = 0.5,
        fluxfrac_metric: float = 0.9,
        target_index: int = 0,
        source_mag: Optional[Union[list[float], np.ndarray]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Calculates an aperture for the user based on the PSF. The user may pick from 
        three options depending on the object of interest and the level of crowding.

        Parameters
        ----------

        aperture_type : str
            A string in which the user can specify the kind of aperture they want 
            calculated.
            - "snr": Based on the cumulative S/N of the target vs other objects in the 
            data cube. Computes the local minima of the cumulative S/N and returns the 
            pixel index for which this occurs. This is then used to create the aperture.
            - "simple": Computed using the target prf model only. Calculates the cumulative 
            relative flux and uses a completness parameter input by the user to 
            determine the aperture.
            - "balanced": This is computes the aperture based on user input crowdsap and 
            flfrcsap values.

        completeness : float
            The relative fraction of flux within a given aperture divided by the total 
            flux of the object. This value is used to compute the simple aperture.
        crowding_metric: Float
            The crowding metric reflects what fraction of the flux in the aperture is 
            due to the target itself not the nearby light sources. Should be flux of 
            source/total flux of everything in the prf data cube.
        fluxfrac_metric: Float
            The flux fraction is similar to excess flux leaking into the aperture, a 
            fraction of the prf of the target may not be captured in it. To account 
            for this missing fraction, the flux fraction is computed.
        target_index: int
            The index of the target within the prf data cube.
        tess_mag: List[float] or np.ndarray
            The Tess magnitudes of all objects within the prf data cube. Magnitudes 
            must be listed in the order present within the data cube.

        Returns
        -------
        A boolean array which can be used as an aperture within lightkurve.

        """
        # in case no mission assigned (e.g. for for the abstract method) we use "generic"
        if not hasattr(self, "mission"):
            self.mission = "generic"

        self.aperture_model = Aperture(
                    model_prf=self.model_prf, source_mag=source_mag, target_index=target_index, mission=self.mission,
                )

        # Want to restrict input of apertures to those allowed
        allowed_apertures = ["snr", "simple", "balanced"]

        if aperture_type not in allowed_apertures:
            raise ValueError(
                            f"User did not enter valid aperture type. Types allowed are {allowed_apertures}"
                        )
        # the simple aperture does not require extra info, only the PRF model for the target.
        if aperture_type == "simple":
            aperture_mask, _ = self.aperture_model.simple_aperture(completeness)
        # to compute SNR optimal aperture we need to make sure we provide the source magnitudes
        elif (aperture_type == "snr") and (source_mag is not None):
            aperture_mask, _ = self.aperture_model.SNR_aperture(**kwargs)
        # balance aperture uses crowding and completeness target, and source magnitudes
        # to get accurate estimates of the metrics.
        elif ((aperture_type == "balanced") and (source_mag is not None)):
            aperture_mask, _ = self.aperture_model.balanced_aperture(
                crowding_metric, fluxfrac_metric
            )
        else:
            raise TypeError(
                f"You must provide relevant arguments for the requested aperture type {aperture_type}. "
                "See documentation for more details."
            )

        return aperture_mask

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
        self.check_coordinates(targets=targets, shape=shape)
        self._prepare_supersamp_prf(targets=targets, shape=shape)
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
        self.date = hdulist[0].read_header()["DATE"]
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

        PRFcol = np.arange(0.5, np.shape(PRFdata[0])[1] + 0.5)
        PRFrow = np.arange(0.5, np.shape(PRFdata[0])[0] + 0.5)

        # Shifts pixels so it is in pixel units centered on 0
        PRFcol = (PRFcol - np.size(PRFcol) / 2) * cdelt1p[0]
        PRFrow = (PRFrow - np.size(PRFrow) / 2) * cdelt2p[0]

        (
            self.PRFrow,
            self.PRFcol,
            self.PRFdata,
            self.crval1p,
            self.crval2p,
            self.cdelt1p,
            self.cdelt2p,
        ) = (PRFrow, PRFcol, PRFdata, crval1p, crval2p, cdelt1p, cdelt2p)

    @abstractmethod
    def check_coordinates(self, targets, shape):
        """Method to check if selected pxels contain collatoral pixels

        Wrap this parent method, use the public method to check that e.g. targets are in bounds.
        Provide a warning if pixels are out of bounds
        """
        pass

    @abstractmethod
    def _prepare_supersamp_prf(self, targets, shape):
        """Method to update the interpolation function

        This method sets up the RectBivariateSpline function to interpolate the supersampled PRF
        """
        pass
