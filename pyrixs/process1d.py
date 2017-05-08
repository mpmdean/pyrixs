""" Functions for processing 1D spectra data
i.e. photon counts versus energy or pixel
Everything is stored in an ordered dictionary
Spectra
 -- spectra -- Pandas dataframe of all spectra
 -- spectrum -- Pandas series representing the sum of all spectra
 -- shifts -- number of pixels that
 -- meta -- A string can be used to pass around additional data.

 Typical workflow is explained in run_rest()
"""

import numpy as np
import pandas as pd
import os, glob
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict

import lmfit

from pyrixs import loaddata

def get_all_spectra_names(search_path):
    """Returns list of file names meeting the folder search term

    Parameters
    ----------
    search_path : string
        wildcard search string e.g. ../folder/*.txt for glob

    Returns
    ---------
    spectra names : list
        all file names matching search_path
    """
    paths = glob.glob(search_path)
    names = [os.path.split(path)[-1] for path in paths]
    return sorted(names)

def load_spectra(search_path, selected_file_names):
    """Load all spectra
    One pandas series will be created per spectrum and stored in a
    pandas dataframe.

    Parameters
    -----------
    search_path : string
        Folder containing files
    selected_file_names : list
        filenames to load

    Returns
    -----------
    spectra : pandas dataframe
        Pandas dataframe of all spectra
    """
    folder = os.path.dirname(search_path)

    spectra = []
    for name in selected_file_names:
        data = loaddata.get_spectrum(os.path.join(folder, name))
        spectrum = pd.Series(data[:,2], index=np.arange(len(data[:,2])), name=name)
        spectra.append(spectrum)

    if spectra != []:
        return pd.concat(spectra, axis=1)
    else:
        print("No spectra loaded")


def make_fake_spectrum(x, elastic_energy=50):
    """Generate intensity values for simulated spectrum.
    One elastic line appears at elastic_energy
    A constant emission line is also added."""
    def peak(x, FWHM=3., center=50., amplitude=10.):
        two_sig_sq = (FWHM / (2 * np.log(2) ) )**2
        return amplitude * np.exp( -(x-center)**2/two_sig_sq )

    y1 = peak(x, FWHM=5., center=elastic_energy, amplitude=3.)
    y2 = peak(x, FWHM=10., center=150., amplitude=10.)
    return y1 + y2 + np.random.rand(len(x))

def make_fake_spectra(elastic_energies=np.arange(40,50)):
    """Return pandas dataframe of simulated spectra"

    Parameters
    -----------
    elastic_energies : array
        array defining position of peak. It is typical for this
        to drift in the course of an experiment.

    Returns
    ----------
    spectra : pandas dataframe
        Pandas dataframe of all spectra
    """
    x = np.arange(200)
    spectra_list = []
    for i, elastic_energy in enumerate(elastic_energies):
        y = make_fake_spectrum(x, elastic_energy=elastic_energy)
        spectrum = pd.Series(y, index=x, name=str(i))
        spectra_list.append(spectrum)

    return pd.concat(spectra_list, axis=1)

def plot_spectra(ax1, spectra, align_min=None, align_max=None):
    """Plot spectra on ax1

    Parameters
    ------------
    ax1 : matplotlib axis object
        axis to plot on
    align_min : float
        vertical line defining minimum alignment value
    align_max : float
        vertical line defining maximum alignment value

    Returns
    -------
    artists : list of matplotlib artist
        from plotting spectra
    xmin_artist : matplotlib artist
        from axvline for min
    xmax_artist : matplotlib artist
        from axvline for max
    """
    plt.sca(ax1)
    while ax1.lines != []:
        ax1.lines[0].remove()

    num_spectra = spectra.shape[1]
    plot_color = iter(matplotlib.cm.spectral(np.linspace(0,1,num_spectra)))
    #for name, spec in spectra.iteritems():
    artists = [spec.plot(color=next(plot_color)) for _, spec in spectra.iteritems()]
    plt.xlabel('Pixel / Energy')
    plt.ylabel('Photons')

    plt.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})

    if align_min is None:
        align_min = spectra.index.min()
    if align_max is None:
        align_max = spectra.index.max()

    xmin_artist, xmax_artist = [plt.axvline(x=x, color='b') for x in [align_min, align_max]]
    return artists, xmin_artist, xmax_artist

def plot_shifts(ax2, shifts):
    """Plot the shift values onto ax2

    Parameters
    ----------
    ax2 : matplotlib axis object

    Returns
    ---------
    shifts_line : matplotlib artist
    """
    plt.sca(ax2)
    try:
        shifts_line.remove()
    except:
        pass
    shifts_line = shifts.plot(style='o')
    plt.ylabel('Shift (pixels)')
    #plt.xticks(Spectra['shifts'].index, [name for name in Spectra['spectra'].columns], rotation=30)
    return shifts_line


def get_shifts(spectra, reference, align_min, align_max, background=0.):
    """Determine the shift required to line up spectra with ref

    Parameters
    ------------
    spectra : pandas dataframe
    reference : pandas series
        Everything is lined up to this.
    align_min : float
        min range of data used in cross-correlation
    align_max : float
        max range of data used in cross-correlation
    background : float
        subtract off this value before cross-correlation (default 0)

    Returns
    ---------
    shifts : pandas series
        shifts indexed by name
    """
    choose_range = np.logical_and(reference.index>align_min, reference.index<align_max)
    ref = reference[choose_range].values

    zero_shift = np.argmax(np.correlate(ref-background, ref-background, mode='Same'))

    shifts = []
    names = []
    for name, spectrum in spectra.iteritems():

        choose_range = np.logical_and(spectrum.index>align_min, spectrum.index<align_max)
        spec = spectrum[choose_range].values

        cross_corr = np.correlate(ref-background, spec-background, mode='Same')
        shift = np.argmax(cross_corr) - zero_shift
        shifts.append(shift)
        names.append(name)

    return pd.Series(data=shifts, index=names)

def get_shifts_w_mean(spectra, reference, align_min, align_max, background=0.):
    """Determine the shift required to line up spectra.
    The mean of the spectra after an initial alignment is used as a reference

    Parameters
    ------------
    reference : pandas series
        reference spectrum
    spectra : pandas dataframe
    align_min : float
        min range of data used in cross-correlation
    align_max : float
        max range of data used in cross-correlation
    background : float
        subtract off this value before cross-correlation (default 0)

    Returns
    ---------
    shifts : pandas series
        shifts indexed by name
    """
    # first alignment and make reference
    shifts = get_shifts(spectra, reference, align_min=align_min, align_max=align_max, background=background)
    reference = apply_shifts(spectra.copy(), shifts).sum(axis=1)

    return get_shifts(spectra, reference, align_min, align_max, background=background)

def apply_shifts(spectra, shifts):
    """ Apply shifts values to spectra and return updated spectra

    Parameters
    ----------
    spectra : pandas dataframe
        Pandas dataframe of all spectra
    shifts : pandas series
        shifts indexed by name

    Returns
    ----------
    spectra : pandas dataframe
        All spectra
    """
    aligned_spectra = []
    for name, spec in spectra.iteritems():
        shifted_intensities = np.interp(spec.index - shifts[name], spec.index, spec.values)
        spec[:] = shifted_intensities
        aligned_spectra.append(spec)

    return pd.concat(aligned_spectra, axis=1)

def sum_spectra(spectra, shifts):
    """Add all the spectra together after alignment

    Parameters
    ----------
    spectra : pandas dataframe
        Pandas dataframe of all spectra

    Returns
    ----------
    spectrum : pandas series
        sum of all spectra
    meta : string
        description of shifts is applied here
    """
    spectrum = spectra.sum(axis=1)

    meta = '# Shifts applied \n'
    for name in spectra.keys():
        meta += '# {}\t {} \n'.format(name, shifts[name])
    #meta += '\n'

    return spectrum, meta

def plot_sum(ax3, spectrum):
    """Plot the summed spectra on ax3

    Parameters
    ------------
    ax3 : matplotlib axis object
        axis to plot on
    spectrum : pandas series
        sum of all spectra

    Returns
    -------
    sum_line : matplotlib artist
        from axvline for min
    """
    plt.sca(ax3)
    sum_line = spectrum.plot()
    plt.xlabel('Pixel / Energy')
    plt.ylabel('Photons')
    return sum_line

def calibrate_spectrum(spectrum, zero_E, energy_per_pixel):
    """Convert spectrum x-axis from pixels to energy

    Parameters
    ----------
    spectrum : pandas series
        sum of all spectra
    zero_E : float
        the pixel corresponding to zero energy loss
    energy_per_pixel : float
        number of meV or eV in one pixel

    Returns
    ---------
    spectrum : pandas series
        sum of all spectra with calibrated indices
    """
    spectrum.index = (spectrum.index-zero_E)*energy_per_pixel
    return spectrum

def save_spectrum(savefilepath, spectrum, meta):
    """Save final spectrum

    Parameters
    ----------
    spectrum : pandas series
        sum of all spectra with calibrated indices
    meta : string
        description of shifts applied
    savefilepath : string
        path to save file at
    """
    f = open(savefilepath, 'w')
    f.write(meta)
    f.write(spectrum.to_string())
    f.close()

def run_test(search_path='test_data/*.txt'):
    """Calls all functions from the command line
    in order to perform a dummy analysis on test_data/*.txt

    Parameters
    -----------
    search_path : string
        search for files following this wildcard

    Returns
    --------
    ax1 : matplotlib axis object
        All spectra
    ax2 : matplotlib axis object
        Shifts applied during lineup
    ax3 : matplotlib axis object
        Summed spectrum before and after calibrating
        which are returned to edit the plots
    """

    # Create plot windows
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    # Load spectra
    spectra = load_spectra(search_path, get_all_spectra_names(search_path))
    if spectra is not None:
        print("Making fake spectra")
        spectra = make_fake_spectra()

    # Align spectra
    align_min = 10
    align_max = 70
    artists = plot_spectra(ax1, spectra, align_min=align_min, align_max=align_max)
    first_spectrum_name = spectra.columns[0]
    shifts = get_shifts(spectra, spectra[first_spectrum_name], align_min=align_min, align_max=align_max, background=0.5)
    #shifts = get_shifts(spectra, spectra[first_spectrum_name], align_min=align_min, align_max=align_max, background=0.5)

    spectra = apply_shifts(spectra, shifts)
    artists = plot_spectra(ax1, spectra, align_min, align_max)

    # Plot the shifts applied
    artists_shifts = plot_shifts(ax2, shifts)

    # Sum, calibrate to energy and plot
    spectrum, meta = sum_spectra(spectra, shifts)
    zero_energy = 40
    energy_per_pixel = 2
    spectrum = calibrate_spectrum(spectrum, zero_energy, energy_per_pixel)
    artists_sum = plot_sum(ax3, spectrum)

    # Save result
    save_spectrum('out.dat', spectrum, meta)

    return ax1, ax2, ax3

if __name__ == "__main__":
    print('Run a test of the code')
    run_test()
