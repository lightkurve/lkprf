"""Class to create apertures for PRF models."""

from typing import List, Optional, Tuple, Union

import numpy as np

from . import logger

ZP_MISSION = {"kepler": 25.132, "k2": 25.132, "tess": 20.44}


class Aperture:
    """Construct an aperture mask from a PRF scene model.

    The aperture is built by ranking pixels according to the flux contribution
    of a target source and then selecting a subset of pixels according to a
    chosen aperture metric. The class stores the scene model, target flux, and
    cumulative metrics used to derive aperture masks.

    Attributes
    ----------
    model_prf : np.ndarray
        Reshaped PRF model array with one source per row.
    source_mag : np.ndarray or None
        Source magnitudes used to compute fluxes.
    target_index : int
        Index of the target source.
    mission : str
        Mission name used for the flux zero-point.
    image_shape : tuple
        Shape of the 2D image represented by a flattened PRF pixel vector.
    scene_flux_cube : np.ndarray
        Flux cube for all sources in the scene.
    target_flux : np.ndarray
        Flux values of the target source at each pixel.
    scene_flux : np.ndarray
        Total scene flux at each pixel.
    sort_index : np.ndarray
        Pixel indices sorted by descending target flux.
    """

    def __init__(
        self,
        model_prf: np.ndarray,
        source_mag: Optional[Union[List[float], np.ndarray]] = None,
        target_index: int = 0,
        mission: str = "TESS",
    ):
        """Initialize an aperture object from a PRF model scene.

        Parameters
        ----------
        model_prf : np.ndarray
            Array of PRF models for one or more sources. The first dimension
            corresponds to the sources and the remaining dimensions define the
            image shape.
        source_mag : array-like of float, optional
            Apparent magnitudes for each source in the same order as `model_prf`.
            If not provided, a default magnitude of 10.0 is assumed for all
            sources.
        target_index : int, default 0
            Index of the source to treat as the target when building the aperture.
        mission : str, default "TESS"
            Mission name used to select the zero-point for converting magnitudes
            to fluxes. Supported values include "kepler", "k2", and "tess".
        """

        # we check if target list is within number of sources
        if target_index > len(model_prf):
            raise ValueError("Provided target index is out of size for `mode_prf`")

        self.model_prf = model_prf.reshape(len(model_prf), -1)
        self.source_mag = source_mag
        self.target_index = target_index
        self.mission = mission

        self.image_shape = model_prf.shape[1:]

        # compute the scene model (cube) in flux units
        self.scene_flux_cube = self._compute_prf_flux()
        # compute the target flux and scene flux
        self.target_flux = self.scene_flux_cube[self.target_index].ravel()
        self.scene_flux = self.scene_flux_cube.sum(axis=0).ravel()
        # keep sort index for target flux
        self.sort_index = np.argsort(self.target_flux)[::-1]

    def _compute_prf_flux(self) -> np.ndarray:
        """Convert PRF models into flux values for each source.

        Returns
        -------
        np.ndarray
            Flux cube with shape ``(n_sources, n_pixels)``.
        """

        if self.source_mag is None:
            logger.warning(
                "Source magnitude not provided, for accurate aperture construction "
                "and metric estimation please provide these values. "
                "Using constant magnitude 10.0 for all sources as default."
            )
            self.source_mag = np.ones(len(self.model_prf), dtype=float) * 10
        # we check if the number of rows in model_prf is the same as provided sources
        if len(self.model_prf) != len(self.source_mag):
            raise ValueError(
                "Number of sources does not match between `model_prf` and `source_mag`"
            )
        if isinstance(self.source_mag, list):
            self.source_mag = np.array(self.source_mag).ravel()

        zpt = ZP_MISSION.get(self.mission.lower(), 20.0)

        # convert Tmag to flux values and dot with PRF model
        tess_flux = 10 ** ((zpt - self.source_mag) / 2.5)
        model_prf_flux = self.model_prf * tess_flux[:, None]

        return np.array(model_prf_flux)

    def _compute_cumulative_FLFRCSAP(self) -> np.ndarray:
        """
        Compute the cumulative flux-fraction metric for the target PRF.
        This metric reports the fraction of the target flux contained in
        the cumulative aperture as pixels are added.

        Returns
        -------
        np.ndarray
            Cumulative flux-fraction values for each increasing aperture size.
        """

        # Create new sorted target array based only on this
        target_flux_sorted = self.target_flux[self.sort_index]

        # compute cumulative sum and flux fraction
        cumulative_FLFRCSAP = np.cumsum(target_flux_sorted) / target_flux_sorted.sum()

        return cumulative_FLFRCSAP

    def _compute_cumulative_CROWDSAP(self) -> np.ndarray:
        """
        Compute the cumulative crowding metric for the target PRF.
        The crowding metric reflects what fraction of the flux in the aperture
        is due to the target itself rather than the nearby light sources.

        Returns
        -------
        np.ndarray
            Fraction of the cumulative aperture flux attributable to the target
            source at each aperture size.
        """

        # Create new sorted target array based only on this
        target_flux_sorted = self.target_flux[self.sort_index]

        # Cumulative sum of this value as if you were adding pixels
        target_cumsum = np.cumsum(target_flux_sorted)

        # Cumulative sum of all signal (target + bkg) sorted by target pixel brightness
        model_prf_cumsum = np.cumsum(self.scene_flux[self.sort_index])

        cumulative_CROWDSAP = target_cumsum / model_prf_cumsum

        return cumulative_CROWDSAP

    def _compute_cumulative_SNR(
        self, read_noise: float = 0.0, quantization_noise: float = 0.0
    ) -> np.ndarray:
        """Compute the cumulative signal-to-noise ratio (SNR) as pixels are added in order
        of highest SNR first.

        Parameters
        ----------
        read_noise : float, default 0.0
            Additional read-noise term in the same flux units as the model. This is
            usually recorded in the target pixel files (TPF) with under 'READNOI{output}'
            with {output} one of the CCD outputs (e.g. TESS CCDs have 4 outputs 'A', 'B',
            'C' and 'D')
        quantization_noise : float, default 0.0
            Additional quantization-noise term in the same flux units as the
            model. This noise is computed as (Smith et al. 2016):
                quant_noise = sqrt(n_c / 12) * (w/ 2 ^(n_b -1)) ** 2
            where n_c is the number of cadences in a co-added observation, w is the
            well depth of the detector, and n_b is the number of bits in the analog-to-digital
            conversion (14 for Kepler and 16 for TESS). The number of cadences is found
            in the 'NREADOUT' header keyword.

        Returns
        -------
        np.ndarray
            Cumulative signal-to-noise ratio values for each increasing
            aperture size.
        """

        # Create new sorted target array based only on this
        target_flux_sorted = self.target_flux[self.sort_index]

        # Cumulative sum of this value as if you were adding pixels to the aperture
        target_cumsum = np.cumsum(target_flux_sorted)

        # compute variance from the scene (all sources) model flux with Poisson noise
        # adding read and quantization noise if provided
        variance = self.scene_flux + read_noise**2 + quantization_noise**2
        noise_cumsum = np.sqrt(np.cumsum(variance[self.sort_index]))

        cumulative_snr = target_cumsum / noise_cumsum

        return cumulative_snr

    def compute_CROWDSAP(self, aperture) -> float:
        """Compute the crowding metric for a given aperture mask.

        Parameters
        ----------
        aperture : array-like
            Boolean mask or weighted aperture definition applied to the
            flattened pixel array.

        Returns
        -------
        float
            Fraction of flux in the supplied aperture due to the target source.
        """

        # Get sum of target flux in aperture
        sum_target_flux = np.sum(self.target_flux * aperture.ravel())

        # Get sum of all flux in aperture
        all_flux = np.sum(self.scene_flux * aperture.ravel())

        Di = sum_target_flux / all_flux

        return Di

    def compute_FLFRCSAP(self, aperture) -> float:
        """Compute the flux-fraction metric for a given aperture mask.

        Parameters
        ----------
        aperture : array-like
            Boolean mask or weighted aperture definition applied to the
            flattened pixel array.

        Returns
        -------
        float
            Fraction of the total target flux contained in the supplied
            aperture.
        """

        # Get sum of target flux in aperture
        sum_target_flux = np.sum(self.target_flux * aperture.ravel())

        # Get sum of all flux in aperture
        flux_frac = sum_target_flux / self.target_flux.sum()

        return flux_frac

    def simple_aperture(self, completeness: float = 0.9) -> Tuple[np.ndarray, float]:
        """Construct an aperture that reaches a target flux completeness.

        Parameters
        ----------
        completeness : float, default 0.9
            Fraction of the target flux that should be included in the
            resulting aperture.

        Returns
        -------
        aperture mask: np.ndarray
            A 2D boolean aperture mask
        Di: float
            Its corresponding crowding metric.
        """

        # Calclate the flux fraction
        FLFRCSAP = self._compute_cumulative_FLFRCSAP()

        # Now we want to select the number of pixels based on the input completeness
        index_pixel = np.where(FLFRCSAP >= completeness)[0][0]

        # Create boolean array which is all false based on the above shape
        aperture_mask = np.full(self.target_flux.shape, False, dtype=bool).flatten()

        # Create masks which are true for index of descending_indices
        indexes_to_set_true = self.sort_index[0 : index_pixel + 1]
        aperture_mask[indexes_to_set_true] = True

        # Calculate Dilution factor
        Di = self.compute_CROWDSAP(aperture_mask)

        return aperture_mask.reshape(self.image_shape), Di

    def SNR_aperture(self, **kwargs) -> Tuple[np.ndarray, float]:
        """Construct an aperture that maximizes the cumulative signal-to-noise.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`_compute_cumulative_SNR`, such as ``read_noise`` and
            ``quantization_noise``.

        Returns
        -------
        aperture mask: np.ndarray
            A 2D boolean aperture mask
        Di: float
            Its corresponding crowding metric.
        """

        cumulative_snr = self._compute_cumulative_SNR(**kwargs)

        # Find the exact number of pixels that maximizes the cumulative SNR
        optimal_pixel_count = np.argmax(cumulative_snr) + 1

        # 6. Create the initial unconstrained optimal mask
        optimal_indices = self.sort_index[:optimal_pixel_count]
        aperture_mask = np.zeros_like(cumulative_snr, dtype=bool)
        aperture_mask[optimal_indices] = True

        # Calculate Dilution factor
        Di = self.compute_CROWDSAP(aperture_mask)

        return aperture_mask.reshape(self.image_shape), Di

    def balanced_aperture(
        self, crowding_metric: float = 0.8, fluxfrac_metric: float = 0.9
    ) -> Tuple[np.ndarray, float]:
        """Construct an aperture that satisfies crowding and flux thresholds.

        Parameters
        ----------
        crowding_metric : float, default 0.8
            Minimum crowding fraction required for inclusion in the aperture.
        fluxfrac_metric : float, default 0.9
            Maximum flux-fraction threshold required for inclusion in the
            aperture.

        Returns
        -------
        aperture mask: np.ndarray
            A 2D boolean aperture mask
        Di: float
            Its corresponding crowding metric.
        """

        # Calculate the cumulative crowdfrac
        crowding = self._compute_cumulative_CROWDSAP()

        # Calculate the cumulative fluxfrac
        flfrac = self._compute_cumulative_FLFRCSAP()

        vals = []
        for a in range(len(crowding)):
            if flfrac[a] < fluxfrac_metric and crowding[a] > crowding_metric:
                vals.append(a)

        shape = self.model_prf.shape[1:]
        aperture_mask = np.full(shape, False, dtype=bool).flatten()
        indexes_to_set_true = self.sort_index[vals]

        for index in indexes_to_set_true:
            aperture_mask[index] = True

        # Calculate Dilution factor
        Di = self.compute_CROWDSAP(aperture_mask)

        return aperture_mask.reshape(self.image_shape), Di
