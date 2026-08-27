import numpy as np


def _get_channel_lookup_array():
    """Returns a lookup table which maps (module, output) onto channel."""
    # In the array below, channel == array[module][output]
    # Note: modules 1, 5, 21, 25 are the FGS guide star CCDs.
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 85, 0, 0, 0],
            [2, 1, 2, 3, 4],
            [3, 5, 6, 7, 8],
            [4, 9, 10, 11, 12],
            [5, 86, 0, 0, 0],
            [6, 13, 14, 15, 16],
            [7, 17, 18, 19, 20],
            [8, 21, 22, 23, 24],
            [9, 25, 26, 27, 28],
            [10, 29, 30, 31, 32],
            [11, 33, 34, 35, 36],
            [12, 37, 38, 39, 40],
            [13, 41, 42, 43, 44],
            [14, 45, 46, 47, 48],
            [15, 49, 50, 51, 52],
            [16, 53, 54, 55, 56],
            [17, 57, 58, 59, 60],
            [18, 61, 62, 63, 64],
            [19, 65, 66, 67, 68],
            [20, 69, 70, 71, 72],
            [21, 87, 0, 0, 0],
            [22, 73, 74, 75, 76],
            [23, 77, 78, 79, 80],
            [24, 81, 82, 83, 84],
            [25, 88, 0, 0, 0],
        ]
    )


def channel_to_module_output(channel):
    """Returns a (module, output) pair given a CCD channel number.

    Parameters
    ----------
    channel : int
        Channel number

    Returns
    -------
    module, output : tuple of ints
        Module and Output number
    """
    if channel < 1 or channel > 88:
        raise ValueError("Channel number must be in the range 1-88.")
    lookup = _get_channel_lookup_array()
    lookup[:, 0] = 0
    modout = np.where(lookup == channel)
    return modout[0][0], modout[1][0]

def saturate_and_bleed(prf_flux: np.ndarray, well_depth: float = 1.5e5) -> np.ndarray:
    """
    Applies CCD saturation and bleed column effects to a PRF flux model.

    Any flux exceeding the CCD well depth is evenly spilled up and down
    its respective column. Flux is conserved within the image footprint.

    Parameters
    ----------
    prf_flux : np.ndarray
        A 2D array representing the evaluated PRF flux model for the scene.
    well_depth : float
        The saturation limit (well depth) of the CCD pixels in flux units.

    Returns
    -------
    np.ndarray
        A 2D array of the saturated flux model containing vertical bleed columns.
    """
    # cap all pixels at well depth and extract the overflow pool
    sat_flux = np.minimum(prf_flux, well_depth)
    excess = np.maximum(prf_flux - well_depth, 0)

    rows, cols = sat_flux.shape

    for c in range(cols):
        col_excess = excess[:, c]

        # skip columns that do not exceed the well depth
        if not np.any(col_excess > 0):
            continue

        col_flux = sat_flux[:, c]

        # find contiguous blocks of saturated pixels
        # Pad with False to catch transitions at the detector edges
        is_sat = col_excess > 0
        padded = np.pad(is_sat, (1, 1), mode="constant", constant_values=False)
        diff = np.diff(padded.astype(int))

        # 'starts' are the top boundaries, 'ends' are the bottom boundaries
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # spill the aggregated excess from the boundaries of saturated star
        for start, end in zip(starts, ends):
            # Sum the total excess for this specific contiguous star
            spill = col_excess[start : end + 1].sum()
            spill_up = spill / 2.0
            spill_down = spill / 2.0

            # spill UP (towards index 0)
            up_idx = start - 1
            while spill_up > 0 and up_idx >= 0:
                capacity = well_depth - col_flux[up_idx]
                if capacity > 0:
                    fill = min(spill_up, capacity)
                    col_flux[up_idx] += fill
                    spill_up -= fill
                up_idx -= 1

            # spill DOWN (towards index rows - 1)
            down_idx = end + 1
            while spill_down > 0 and down_idx < rows:
                capacity = well_depth - col_flux[down_idx]
                if capacity > 0:
                    fill = min(spill_down, capacity)
                    col_flux[down_idx] += fill
                    spill_down -= fill
                down_idx += 1

        # set the modified column back into the array
        sat_flux[:, c] = col_flux

    return sat_flux

class LKPRFWarning(Warning):
    """Class for lkprf warnings."""

    pass
