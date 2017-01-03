""" Methods for loading data.
This is separated for process1d and process2d in order to facilitate loading
data from different sources.
"""

import numpy as np
import h5py

def getimage(filename):
    """Return list of photon events as XY pairs
    shape rows, 2

    files with .h5 are assumed to be SLS format
    """
    if filename[-3:] == '.h5':
        h5file = h5py.File(filename)
        XY = h5file['entry']['analysis']['events'].value
        return XY

def getspectrum(filename):
    """ return spectrum as
    pixel intensity
    shape rows, 2

    .txt are compatible with np.loadtxt
    """
    if filename[-4:] == '.txt':
        return np.loadtxt(filename)
