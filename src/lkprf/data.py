"""Module to get the archived data for PRFs"""

import os  # noqa
import requests
import numpy as np
import fitsio
from scipy.ndimage import label, uniform_filter

from . import logger, PACKAGEDIR

__all__ = [
    "download_kepler_prf_file",
    "get_kepler_prf_file",
    "build_tess_prf_file",
    "get_tess_prf_file",
    "clear_kepler_cache",
    "clear_tess_cache",
]


_tess_prefixes = {
    1: {
        1: "2019107181900",
        2: "2019107181901",
        3: "2019107181901",
        4: "2019107181901",
    },
    2: {
        1: "2019107181901",
        2: "2019107181901",
        3: "2019107181901",
        4: "2019107181901",
    },
    3: {
        1: "2019107181901",
        2: "2019107181902",
        3: "2019107181902",
        4: "2019107181902",
    },
    4: {
        1: "2019107181902",
        2: "2019107181902",
        3: "2019107181902",
        4: "2019107181902",
    },
}


def _download_file(url, file_path):
    """
    Download the required data files to the specified directory.
    """
    if not os.path.exists(file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            response = requests.get(url)
            response.raise_for_status()  # Ensure the request was successful
            with open(file_path, "wb") as f:
                f.write(response.content)
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404:
                return
            else:
                raise http_err


def download_kepler_prf_file(module: int, output: int):
    """Download a Kepler Module file"""
    filename = f"kplr{module:02}.{output}_2011265_prf.fits"
    file_path = f"{PACKAGEDIR}/data/{filename}"
    url = f"https://archive.stsci.edu/missions/kepler/fpc/prf/{filename}"
    logger.info(f"Downloading {module:02}.{output}")
    _download_file(url, file_path)
    hdulist = fitsio.FITS(file_path)
    for hdu in hdulist:
        hdu.verify_checksum()
    return


def build_tess_prf_file(camera: int, ccd: int):
    """Download a set of TESS PRF files for a given camera/ccd"""

    def open_file(url: str):
        """Open a TESS PRF file and correct the edges to 0"""
        file_name = url.split("/")[-1]
        _download_file(url, f"/tmp/{file_name}")
        hdulist = fitsio.FITS(f"/tmp/{file_name}")
        data = hdulist[0].read()
        hdr = hdulist[0].read_header()
        aper = data > 0.0005  # cutout_model > np.percentile(prf, 50)
        labels = label(aper)[0]  # assign group labels to the image
        psf_region = np.bincount(
            labels[aper].flatten()
        ).argmax()  # find the biggest region that contains the prf core
        mask = uniform_filter(
            labels != psf_region, size=10
        )  # apply a filter to smooth out the region
        data[mask] = 0
        return data, hdr

    tess_archive_url = (
        "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0004/"
    )
    prefix = _tess_prefixes[camera][ccd]
    filename = f"tess-prf-{camera}-{ccd}.fits"
    file_path = f"{PACKAGEDIR}/data/{filename}"
    # ensure the file_path exists
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    R, C = np.meshgrid([1, 513, 1025, 1536, 2048], [45, 557, 1069, 1580, 2092])
    if os.path.isfile(file_path):
        os.remove(file_path)

    hdulist = fitsio.FITS(file_path, mode="rw")
    for (
        r,
        c,
    ) in zip(R.ravel(), C.ravel()):
        url = f"{tess_archive_url}cam{camera}_ccd{ccd}/tess{prefix}-prf-{camera}-{ccd}-row{r:04}-col{c:04}.fits"
        logger.info(f"Downloading CCD {ccd}, Camera {camera}, row {r}, column {c}")
        data, hdr = open_file(url=url)
        hdulist.write(data, header=hdr)
    hdulist.close()
    return


def get_kepler_prf_file(module: int, output: int):
    """Download a Kepler Module file"""
    filename = f"kplr{module:02}.{output}_2011265_prf.fits"
    file_path = f"{PACKAGEDIR}/data/{filename}"
    if not os.path.isfile(file_path):
        logger.info(
            f"No local files found, building Kepler PRF for Module {module}, output {output}."
        )
        download_kepler_prf_file(module=module, output=output)
    file_path = f"{PACKAGEDIR}/data/{filename}"
    hdulist = fitsio.FITS(file_path)
    return hdulist


def get_tess_prf_file(camera: int, ccd: int):
    """Get a PRF file for a given camera/ccd"""
    filename = f"tess-prf-{camera}-{ccd}.fits"
    file_path = f"{PACKAGEDIR}/data/{filename}"
    if not os.path.isfile(file_path):
        logger.info(
            f"No local files found, building TESS PRF for Camera {camera}, CCD {ccd}."
        )
        build_tess_prf_file(camera=camera, ccd=ccd)
    hdulist = fitsio.FITS(file_path)
    return hdulist


def clear_kepler_cache():
    for module in np.arange(26):
        for output in np.arange(1, 5):
            filename = f"kplr{module:02}.{output}_2011265_prf.fits"
            file_path = f"{PACKAGEDIR}/data/{filename}"
            if os.path.isfile(file_path):
                os.remove(file_path)


def clear_tess_cache():
    for camera in np.arange(1, 5):
        for ccd in np.arange(1, 5):
            filename = f"tess-prf-{camera}-{ccd}.fits"
            file_path = f"{PACKAGEDIR}/data/{filename}"
            if os.path.isfile(file_path):
                os.remove(file_path)
