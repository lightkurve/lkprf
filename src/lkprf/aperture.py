"""Class to create apertures for PRF models"""

from typing import Tuple, List
import numpy as np

from .utils import LKPRFWarning
from .data import get_tess_prf_file
from . import PACKAGEDIR
import warnings

from scipy.interpolate import RectBivariateSpline
from scipy.signal import argrelextrema

class aperture:

    def __init__(
        self,
        model_prf = [],  #Must be an 3D array
        target_index: int = 0,  # The index that the target prf is stored in
        tess_mag = list[float]) -> float:  # Must match length of self[0:] and must be an array  

        self.model_prf = model_prf
        self.tess_mag = tess_mag
        self.target_index = target_index

    def _compute_prf_flux(self):
        """Converts the prf model(s) into flux using input tess magnitudes
        """

        zpt = 20.44

        # Creates empty areay to store flux model values
        model_prf_flux = []
        for a in range(len(self.tess_mag)):
            tess_flux = 10 ** ((zpt - self.tess_mag[a]) / 2.5)
            model_prf_flux.append(self.model_prf[a] * tess_flux)

        return np.array(model_prf_flux)

    def _compute_cumulative_FLFRCSAP(self):
        """This will calculate the FLFRCSAP for an aperture that starts from the brightest pixel in the target PRF and then includes the second brightest
        and so on in decreasing value. The flux fraction is similar to excess flux leaking into the aperture, a fraction of the PRF of the target may not
        be captured in it. To account for this missing fraction, the flux fraction is computed."""

        # Grab only the target from the prf cube and flatten the data
        target_flatten = self.model_prf[self.target_index].flatten()

        # Next sort by brightest to faintest
        sort_index = np.argsort(target_flatten)

        # Now decending values
        descending_indices = sort_index[::-1]

        # Create new sorted target array based only on this
        target_flatten_decending = target_flatten[descending_indices]

        # Cumulative sum of this value as if adding pixels to the aperture
        target_cumsum = np.cumsum(target_flatten_decending)

        # Divite the target flux within the aperture by the total flux
        cumulative_FLFRCSAP = target_cumsum / target_cumsum[-1:]

        return cumulative_FLFRCSAP, descending_indices

    def _compute_cumulative_CROWDSAP(self):
        """This will calculate the CROWDSAP for an aperture that starts from the brightest pixel in the
        target PRF and then includes the second brightest and so on in decreasing value.

        The crowding metric reflects what fraction of the flux in the aperture is due to the target itself
        not the nearby light sources. Should be flux of source/total flux of everything."""

        # Need to convert into flux first
        model_prf_flux = self._compute_prf_flux()

        # Grab the target from the prf cube and flatten the data
        target_flatten = model_prf_flux[self.target_index].flatten()

        # Sort by brightest to faintest
        sort_index = np.argsort(target_flatten)

        # Now decending values
        descending_indices = sort_index[::-1]

        # Create new sorted target array based only on this
        target_flatten_decending = target_flatten[descending_indices]

        # Cumulative sum of this value as if you were adding pixels to the aperture
        target_cumsum = np.cumsum(target_flatten_decending)

        # Add up flux for each pixel for every object in whole cube
        model_prf_cube_sum = np.sum(model_prf_flux, axis=0)

        # flatten
        model_prf_cube_sum_flatten = model_prf_cube_sum.flatten()

        # Cumulative sum on index from above
        model_prf_cumsum = np.cumsum(model_prf_cube_sum_flatten[descending_indices])

        cumulative_CROWDSAP = target_cumsum / model_prf_cumsum

        return cumulative_CROWDSAP

    def _compute_cumulative_signaltonoise(self):
        """This will calculate the S/N for an aperture that starts from the brightest pixel in the target
        PRF and then includes the second brightest and so on in decreasing value.
        The flux fraction is similar to excess flux leaking into the aperture, a fraction of the PRF of the
        target may not be captured in it.
        To account for this missing fraction, the flux fraction is computed."""

        # Need to convert into flux first
        model_prf_flux = self._compute_prf_flux()

        # Grab only the target from the prf cube and flatten the data
        target_flatten = model_prf_flux[self.target_index].flatten()

        # Sort by brightest to faintest
        sort_index = np.argsort(target_flatten)

        # Decending values
        descending_indices = sort_index[::-1]

        # Create new sorted target array based only on this
        target_flatten_decending = target_flatten[descending_indices]

        # Cumulative sum of this value as if you were adding pixels to the aperture
        target_cumsum = np.cumsum(target_flatten_decending)

        noise = model_prf_flux[1:]
        noise_sum = np.sum(noise, axis=0)
        noise_flatten = noise_sum.flatten()
        noise_cumsum = np.cumsum(noise_flatten[descending_indices])

        cumulative_SN = target_cumsum / noise_cumsum

        return cumulative_SN, descending_indices

    def _get_local_minima(self,cumulative_SN):
        """This code computes the local minima of the cumulative signal to noise."""

        # Find indices of local minima
        minima_indices = argrelextrema(cumulative_SN, np.less)

        # It is likely there will be a large drop at 1 pixel as pixel 0 will contain the most flux
        # We dont want 1 as the firt minima, as such removing
        minima_indices_fix = np.array(
            tuple(item for item in minima_indices[0] if item != 1)
        )

        return minima_indices_fix[0]


    def _calculate_dilution_factor(self,aperture):
        #Calculates the dilution factor as a function of the aperture selected

        # Convert into flux model
        model_prf_flux = self._compute_prf_flux()

        # Get sum of target flux in aperture
        target_flux = model_prf_flux[self.target_index]
        sum_target_flux = np.sum(model_prf_flux[self.target_index] * aperture)

        # Get sum of all flux in aperture
        all_flux = np.sum(model_prf_flux * aperture)

        Di = sum_target_flux/all_flux

        return Di
    
    def simple_aperture(self, completeness: float = 0.9):

        # Calclate the flux fraction
        FLFRCSAP, descending_indices = self._compute_cumulative_FLFRCSAP()

        # Now you want to select the number of pixels based on the input completness from the user
        index_pixel = np.where(FLFRCSAP >= completeness)
        index_pixel = index_pixel[0][0]

        # Get the initial data so you can determine correct size
        target_data = self.model_prf[self.target_index]

        # Create boolean array which is all false based on the above shape
        all_false_array = np.full(target_data.shape, False, dtype=bool).flatten()

        # Create masks which are true for index of descending_indices
        indexes_to_set_true = descending_indices[0:index_pixel + 1]

        for index in indexes_to_set_true:
            all_false_array[index] = True

        # Re-shape the array into what it was before so we can see what the mask looks like
        simple_aperture = all_false_array.reshape(target_data.shape)

        #Calculate Dilution factor
        Di = self._calculate_dilution_factor(simple_aperture)
        
        
        return simple_aperture, Di

    def strict_aperture(self):

        # Convert into flux model
        model_prf_flux = self._compute_prf_flux()

        # Calculate the cumulative SN
        cumulative_SN, descending_indices = self._compute_cumulative_signaltonoise()

        # Calculate the minima
        minima_indices = self._get_local_minima(cumulative_SN)

        # Get the initial data so you can determine correct size
        target_data = self.model_prf[self.target_index]

        # Create boolean array which is all false based on the above shape
        all_false_array = np.full(target_data.shape, False, dtype=bool).flatten()

        # Create masks which are true for index of descending_indices
        indexes_to_set_true = descending_indices[0: minima_indices + 1]

        for index in indexes_to_set_true:
            all_false_array[index] = True

        # Re-shape the array into what it was before so we can see what the mask looks like
        strict_aperture = all_false_array.reshape(target_data.shape)

        #Calculate Dilution factor
        Di = self._calculate_dilution_factor(strict_aperture)

        return strict_aperture, Di
    

    def balanced_aperture(
        self, crowding_metric: float = 0.8, fluxfrac_metric: float = 0.9
    ):

        # Convert model into flux model
        model_prf_flux = self._compute_prf_flux()

        # Calculate the cumulative crowdfrac
        crowding = self._compute_cumulative_CROWDSAP()

        # Calculate the cumulative fluxfrac
        flfrac, idx = self._compute_cumulative_FLFRCSAP()

        vals = []
        for a in range(len(crowding)):
            if flfrac[a] < fluxfrac_metric and crowding[a] > crowding_metric:
                vals.append(a)

        shape = self.model_prf.shape[1:]
        all_false_array = np.full(shape, False, dtype=bool).flatten()
        indexes_to_set_true = idx[vals]

        for index in indexes_to_set_true:
            all_false_array[index] = True

        # Re-shape the array into what it was before so we can see what the mask looks like
        balanced_aperture = all_false_array.reshape(shape)

        #Calculate Dilution factor
        Di = self._calculate_dilution_factor(balanced_aperture)
        
        return balanced_aperture, Di
