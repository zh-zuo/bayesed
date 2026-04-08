Quickstart
==========

Reading photometry from a CSV file
-----------------------------------

The main entry point for loading data is :func:`bayesed.io.read_photometry_csv`.
It expects a CSV file with at least three columns: ``band``, ``flux``, and ``flux_err``.

.. code-block:: python

   from bayesed.io import read_photometry_csv

   phot = read_photometry_csv("my_photometry.csv")

   print(phot.band)       # array of filter names, e.g. ['F1500W', 'F1800W']
   print(phot.flux)       # array of flux values
   print(phot.flux_err)   # array of flux uncertainties

Example CSV format
------------------

Your CSV file should look something like this:

.. code-block:: text

   band,flux,flux_err,mjd
   F1500W,1.23e-3,1.0e-4,59000.1
   F1800W,2.34e-3,1.2e-4,59000.1
   F2100W,2.10e-3,1.1e-4,59000.2

The ``mjd`` and ``zp`` columns are optional.

Custom column names
-------------------

If your CSV uses different column names, you can specify them:

.. code-block:: python

   phot = read_photometry_csv(
       "data.csv",
       band_col="filter",
       flux_col="f_nu",
       fluxerr_col="f_nu_err",
   )

The Photometry object
---------------------

:class:`bayesed.io.Photometry` is an immutable dataclass that holds your photometric data.
It validates that all arrays have consistent lengths on creation.

.. code-block:: python

   import numpy as np
   from bayesed.io import Photometry

   phot = Photometry(
       band=np.array(["g", "r", "i"]),
       flux=np.array([1.0, 2.0, 3.0]),
       flux_err=np.array([0.1, 0.2, 0.3]),
   )
