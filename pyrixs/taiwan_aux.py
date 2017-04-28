import numpy as np
import matplotlib.pyplot as plt
import os
from pyrixs import process1d,process2d
import pandas as pd
import glob
from scipy.interpolate import interp1d

def load_spectra(scans,folder):
    """Load all spectra. Data scan will either be 1 file saved in folder, or will be
    multiple scans saved in a subfolder labeled with the same name as the scan. This
    will just create the appropriate scan_path input for process1d.load_spectra.

    Parameters
    -----------
    scans : list
        List of scan names.
    folder:
        Folder that contains scans.

    Returns
    -----------
    spectra : pandas dataframe
        Pandas dataframe of all spectra
    """
    
    scan_path = []
    for scan in scans:
        if os.path.isfile(folder + scan + '_XAS.dat') is True:
            scan_path.append(folder + scan + '_XAS.dat')
        elif os.path.isdir(folder + scan) is True:
            scan_path += glob.glob(folder + scan + '/' + scan + '_*_XAS.dat')
        else:
            print('##############################')
            print('## Could not find scan ' + scan + '! ##')
            print('##############################')
            
    return process1d.load_spectra('',scan_path)

def get_shifts_interp(spectra, reference = 'first', align_min = 0, align_max = 2000,
                   background=0., average = 1):
    """Determine the shift required to line up spectra with first or last spectrum.
    It interpolates the data to reduce the pixel size a factor of 10, and averages
    consecutive spectra to improve statistics.
    
    Parameters
    ------------
    spectra : pandas dataframe
        spectra to be aligned
    reference : string
        options are 'first' and 'last'
    align_min : float
        min range of data used in cross-correlation
    align_max : float
        max range of data used in cross-correlation
    background : float
        subtract off this value before cross-correlation (default 0)
    average : int
        number to scans to average to get the shifts
    Returns
    ---------
    shifts : pandas series
        shifts indexed by name, it will have the same number of scans as spectra.
    """
    
    partial_sum_spectra = []
    for i in range(spectra.shape[1]//average):
        scans = spectra.keys()[average*i:average*(i+1)]
        tmp,_ = process1d.sum_spectra(spectra[scans])
        partial_sum_spectra.append(tmp)
    
    new_spectra = pd.concat(partial_sum_spectra,axis=1)
    
    if reference == 'last':
        oldref = new_spectra[new_spectra.keys()[-1]]
    else:
        oldref = new_spectra[new_spectra.keys()[0]]
    
    choose_range = np.logical_and(oldref.index>align_min, oldref.index<align_max)
    oldref = oldref[choose_range].values
    
    newx = np.linspace(0,oldref.size-1,oldref.size*10)
    ref = interp1d([i for i in range(oldref.size)],oldref,kind='linear')(newx)

    zero_shift = np.argmax(np.correlate(ref-background, ref-background, mode='Same'))
    
    partial_shifts = []
    for name, spectrum in new_spectra.iteritems():
        choose_range = np.logical_and(spectrum.index>align_min, spectrum.index<align_max)
        oldspec = spectrum[choose_range].values
        
        spec = interp1d([i for i in range(oldspec.size)],oldspec,kind='linear')(newx)

        cross_corr = np.correlate(ref-background, spec-background, mode='Same')
        shift = np.argmax(cross_corr) - zero_shift
        partial_shifts.append(shift/10.)
        
    
    shifts = [0.0 for i in range(spectra.shape[1])]
    names = spectra.keys()
    for i in range(spectra.shape[1]//average):
        shifts[average*i:average*(i+1)] = [partial_shifts[i] for j in range(average)]
    
    if spectra.shape[1]%average != 0:
        shifts[-(spectra.shape[1]%average):] = [partial_shifts[spectra.shape[1]//average-1] for i in range(spectra.shape[1]%average)]
        
    return pd.Series(data=shifts, index=names)


def partial_sum_spectra(spectra,average):
    """Add spectra in 'average' blocks. This is useful to check
    if the alignment using average != 1 worked well.

    Parameters
    ----------
    spectra : pandas dataframe
        Pandas dataframe of all spectra
        
    average : int
        number of scans to be averaged

    Returns
    ----------
    new_spectra : pandas dataframe
        partial sum of spectra
    new_errors : pandas dataframe
        partial error of spectra calculated based on the spectra standart deviation
    """
    
    # If spectra.shape[1] is not a multiple of avg the last few scans are ignored.
    partial_sum_spectra = []
    partial_sum_errors = []
    new_names = []
    
    for i in range(spectra.shape[1]//average):
        scans = spectra.keys()[average*i:average*(i+1)]
        tmp = spectra[scans]
        partial_sum_spectra.append(tmp.sum(axis=1))
        partial_sum_errors.append(tmp.std(axis=1)*np.sqrt(average))
        new_names.append('{:d}-{:d}'.format(average*i,average*(i+1)-1))
                         
    new_spectra = pd.concat(partial_sum_spectra,axis=1)
    new_spectra.columns = new_names
    
    new_errors = pd.concat(partial_sum_errors,axis=1)
    new_errors.columns = new_names
                         
    return new_spectra,new_errors 


def load_fit_carbon_tape(scans,folder,xmin=1,xmax=2000):
    """Process the carbon tape data.

    Parameters
    ----------
    scans : list
        List of scans to be averaged
    folder : string
        Data folder
    xmin : int
        Low limit on pixel range to be used
    xmax : int
        High limit on pixel range to be used

    Returns
    ----------
    spec : numpy.array
        Carbon tape data
    resolution : list
        List of fitted parameters from process2d.fit_resolution
    """
    
    spectra = load_spectra(scans,folder)
    spectrum,_ = process1d.sum_spectra(spectra)

    spec = np.empty((len(spectrum.index),2))
    spec[:,0] = spectrum.index.values
    spec[:,1] = spectrum.values

    resolution = process2d.fit_resolution(spec,xmin=xmin,xmax=xmax)

    center = resolution[1]
    fwhm = resolution[0]

    print('######################')
    print('## Center at {:0.2f} ##'.format(center))
    print('## FWHM {:0.2f}        ##'.format(fwhm))
    print('######################')
    
    return spec,resolution
    