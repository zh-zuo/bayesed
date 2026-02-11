# bayesed/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Photometry:
    """Container for photometric measurements."""
    band: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    mjd: Optional[np.ndarray] = None
    zp: Optional[np.ndarray] = None  # optional zeropoint, if you store it

    def __post_init__(self):
        # Basic shape checks
        n = len(self.band)
        if len(self.flux) != n or len(self.flux_err) != n:
            raise ValueError("band/flux/flux_err must have the same length.")
        if self.mjd is not None and len(self.mjd) != n:
            raise ValueError("mjd must match length of band.")
        if self.zp is not None and len(self.zp) != n:
            raise ValueError("zp must match length of band.")


def read_photometry_csv(
    path: str | Path,
    *,
    band_col: str = "band",
    flux_col: str = "flux",
    fluxerr_col: str = "flux_err",
    mjd_col: str = "mjd",
    zp_col: str = "zp",
    required: Sequence[str] = ("band", "flux", "flux_err"),
) -> Photometry:
    """
    Read photometry from a CSV file.

    Expected columns by default:
      - band (string), flux (float), flux_err (float)
    Optional:
      - mjd, zp

    Returns
    -------
    Photometry
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Map "required" logical names to actual column names used for this call
    name_map = {
        "band": band_col,
        "flux": flux_col,
        "flux_err": fluxerr_col,
        "mjd": mjd_col,
        "zp": zp_col,
    }

    missing = [k for k in required if name_map[k] not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. "
            f"Have columns={list(df.columns)}"
        )

    band = df[band_col].astype(str).to_numpy()
    flux = pd.to_numeric(df[flux_col], errors="raise").to_numpy(dtype=float)
    flux_err = pd.to_numeric(df[fluxerr_col], errors="raise").to_numpy(dtype=float)

    mjd = df[mjd_col].to_numpy(dtype=float) if mjd_col in df.columns else None
    zp = df[zp_col].to_numpy(dtype=float) if zp_col in df.columns else None

    # Some practical sanity checks
    if np.any(~np.isfinite(flux)) or np.any(~np.isfinite(flux_err)):
        raise ValueError("Found non-finite values in flux/flux_err.")
    if np.any(flux_err <= 0):
        raise ValueError("All flux_err must be > 0.")

    return Photometry(band=band, flux=flux, flux_err=flux_err, mjd=mjd, zp=zp)
