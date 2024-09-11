"""Test the PRFs"""

import lkprf
import numpy as np
import matplotlib.pyplot as plt

import os


def is_github_actions():
    return os.getenv("GITHUB_ACTIONS") == "true"


def test_prfs():
    """Test you can make a prf and it's fairly well centered"""
    for prf in [lkprf.KeplerPRF(channel=42), lkprf.TESSPRF(camera=1, ccd=1)]:
        targets = [(10, 10)]
        origin = (0, 0)
        shape = (21, 21)
        ar = prf.evaluate(targets=targets, origin=origin, shape=shape)
        R, C = np.mgrid[: ar.shape[1], : ar.shape[2]] + 1.0
        assert np.isclose(np.average(R.ravel(), weights=ar[0].ravel()), targets[0][1], atol=0.15)
        assert np.isclose(np.average(C.ravel(), weights=ar[0].ravel()), targets[0][0], atol=0.15)
        assert np.isclose(ar[0].sum(), 1)
        assert ar.shape == (1, *shape)
        if not is_github_actions():
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(
                np.arange(origin[1], origin[1] + shape[1]) + 0.5,
                np.arange(origin[0], origin[0] + shape[0]) + 0.5,
                ar.sum(axis=0),
                cmap="Greys_r",
                vmin=0,
                vmax=0.2,
            )
            cbar = plt.colorbar(im, ax=ax)
            # ax.scatter(np.asarray(targets)[:, 1], np.asarray(targets)[:, 0], c="r")
            ax.set(
                xlim=(origin[1], origin[1] + shape[1] - 1),
                ylim=(origin[0], origin[0] + shape[0] - 1),
                title=f"{prf.mission} PRF Example",
                aspect="equal",
                xlabel="Column [pixel]",
                ylabel="Row [pixel]",
            )
            cbar.set_label("Normalized Flux")
            fig.savefig(f"docs/images/{prf.mission}.png", dpi=250)
            plt.close("all")

        ar = prf.evaluate(targets=(20, 20), origin=origin, shape=shape)
        assert not np.isclose(ar[0].sum(), 1)
        assert not np.isclose(ar[0].sum(), 0)

        ar = prf.gradient(targets=targets, origin=origin, shape=shape)
        assert isinstance(ar, tuple)
        assert ar[0].shape == (1, *shape)
        for a in ar:
            assert (a[0] < 0).any()
            assert (a[0] > 0).any()
