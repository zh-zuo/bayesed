# tests/test_photometry_io.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesed.io import read_photometry_csv


def test_read_photometry_csv_roundtrip(tmp_path):
    # Make a tiny, realistic photometry CSV
    df = pd.DataFrame(
        {
            "mjd": [59000.1, 59000.1, 59000.2],
            "band": ["F1500W", "F1800W", "F2100W"],
            "flux": [1.23e-3, 2.34e-3, 2.10e-3],         # e.g., Jy
            "flux_err": [1.0e-4, 1.2e-4, 1.1e-4],
            "zp": [None, None, None],                   # optional column is allowed
        }
    )
    path = tmp_path / "photometry.csv"
    df.to_csv(path, index=False)

    phot = read_photometry_csv(path)

    assert phot.band.shape == (3,)
    assert phot.flux.shape == (3,)
    assert phot.flux_err.shape == (3,)

    assert phot.band[0] == "F1500W"
    assert np.isclose(phot.flux[1], 2.34e-3)
    assert np.isclose(phot.flux_err[2], 1.1e-4)

    assert phot.mjd is not None
    assert np.isclose(phot.mjd[0], 59000.1)


def test_read_photometry_csv_missing_required_column(tmp_path):
    df = pd.DataFrame({"band": ["g"], "flux": [1.0]})  # missing flux_err
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        read_photometry_csv(path)


def test_read_photometry_csv_rejects_nonpositive_fluxerr(tmp_path):
    df = pd.DataFrame({"band": ["g"], "flux": [1.0], "flux_err": [0.0]})
    path = tmp_path / "bad2.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="flux_err"):
        read_photometry_csv(path)
