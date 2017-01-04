""" Functions for processing 1D spectra data
i.e. photon counts versus energy or pixel
Everything is stored in an ordered dictionary
Spectra
 -- spectra -- Pandas dataframe of all spectra
 -- total_sum -- Pandas series representing the sum of all spectra
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

Spectra = OrderedDict({
'spectra' : pd.DataFrame([]),
'total_sum' : pd.Series([]),
'shifts' : pd.Series([]),
'meta' : ''
})

def get_all_spectra_names(search_path):
    """Returns list of file names meeting the folder search term"""
    paths = glob.glob(search_path)
    return [path.split('/')[-1] for path in paths]

def load_spectra(search_path, selected_file_names):
    """Load all spectra
    One pandas series will be created per spectrum and stored as
    pandas dataframe Spectra['spectra']

    Arguments:
    search_path -- string that defines folder containing files
    selected_file_names -- list of filenames to load
    """
    global Spectra

    folder = os.path.dirname(search_path)

    spectra = []
    for name in selected_file_names:
        data = loaddata.getspectrum(os.path.join(folder, name))
        spectrum = pd.Series(data[:,2], index=np.arange(len(data[:,2])), name=name)
        spectra.append(spectrum)

    if spectra != []:
        Spectra['spectra'] = pd.concat(spectra, axis=1)
    else:
        Spectra['spectra'] = pd.DataFrame([])
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

    Arguments:
    elastic_energies --  array defining position of peak. It is typical for this
    to drift in the course of an experiment.
    """
    x = np.arange(200)
    spectra_list = []
    for i, elastic_energy in enumerate(elastic_energies):
        y = make_fake_spectrum(x, elastic_energy=elastic_energy)
        spectrum = pd.Series(y, index=x, name=str(i))
        spectra_list.append(spectrum)

    return pd.concat(spectra_list, axis=1)

def plot_spectra(ax1, align_min=None, align_max=None):
    """Plot spectra on ax1

    Arguments:
    ax1 -- axis to plot on
    align_min -- vertical line defining minimum alignment value
    align_max -- vertical line defining maximum alignment value
    """
    plt.sca(ax1)
    while ax1.lines != []:
        ax1.lines[0].remove()

    num_spectra = Spectra['spectra'].shape[1]
    plot_color = iter(matplotlib.cm.spectral(np.linspace(0,1,num_spectra)))
    for name, spectrum in Spectra['spectra'].iteritems():
        spectrum.plot(color=next(plot_color))
    plt.xlabel('Pixel / Energy')
    plt.ylabel('Photons')

    plt.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})

    if align_min == None:
        align_min = Spectra['spectra'].index.min()
    if align_max == None:
        align_max = Spectra['spectra'].index.max()
    plt.axvline(x=align_min, color='b')
    plt.axvline(x=align_max, color='b')

def plot_shifts(ax2):
    """Plot the shift values onto ax2"""
    plt.sca(ax2)
    try:
        shifts_line.remove()
    except:
        pass
    shifts_line = plt.plot(Spectra['shifts'], 'ko-', label='Pixel offsets')
    plt.ylabel('Shift (pixels)')
    plt.xticks(Spectra['shifts'].index, [name for name in Spectra['spectra'].columns], rotation=30)


def correlate(ref, spectra, align_min, align_max, background=0.):
    """Determine the shift in pandas index value required to line up spectra with ref

    Arguments:
    spectra -- pandas dataframe containing spectra
    align_min, align_max -- define range of data used in cross-correlation
    background -- subtract off this value before cross-correlation (default 0)

    Returns:
    shifts -- list of shifts
    """
    zero_shift = np.argmax(np.correlate(ref-background, ref-background, mode='Same'))
    xref = np.arange(len(ref))
    shifts = []
    for name, spectrum in spectra.iteritems():

        choose_range = np.logical_and(spectrum.index>align_min, spectrum.index<align_max)
        spec = spectrum[choose_range].values

        cross_corr = np.correlate(ref-background, spec-background, mode='Same')
        shift = np.argmax(cross_corr) - zero_shift
        shifts.append(shift)

    return shifts

def get_shifts(ref_name, align_min, align_max, background=0):
    """Get shifts using cross correlation.

    Arguments
    ref_name -- name of spectrum for zero energy shift
    align_min, align_max -- define range of data used in correlation
    background -- subtract off this value before cross-correlation (default 0)

    Returns:
    shifts -- list of shifts
    """
    ref_spectrum = Spectra['spectra'][ref_name]
    choose_range = np.logical_and(ref_spectrum.index>align_min, ref_spectrum.index<align_max)
    ref = ref_spectrum[choose_range].values

    return correlate(ref, Spectra['spectra'], align_min, align_max, background=background)

def get_shifts_w_mean(ref_name, align_min, align_max, background=0.):
    """Get shifts using cross correlation.
    The mean of the spectra after an initial alignment i used as a reference

    Arguments
    ref_name -- name of spectrum for zero energy shift
    align_min, align_max -- define range of data used in correlation
    background -- subtract off this value before cross-correlation (default 0)

    Returns:
    shifts -- list of shifts
    """
    # first alignment
    shifts = get_shifts(ref_name, align_min, align_max, background=background)

    # do a first pass alignment
    first_pass_spectra = []
    spectra = Spectra['spectra'].copy()
    for shift, (name, spectrum) in zip(shifts, spectra.items()):
        shifted_intensities = np.interp(spectrum.index - shift, spectrum.index, spectrum.values)
        spectrum[:] = shifted_intensities
        first_pass_spectra.append(spectrum)

        ref_spectrum = pd.concat(first_pass_spectra, axis=1).mean(axis=1)

        choose_range = np.logical_and(ref_spectrum.index>align_min, ref_spectrum.index<align_max)
        ref = ref_spectrum[choose_range].values
        return correlate(ref, Spectra['spectra'], align_min, align_max, background=background)

def apply_shifts(shifts):
    """ Apply shifts values to Spectras['spectra'] and update Spectras['shifts']"""
    global Spectra
    aligned_spectra = []
    for shift, (name, spectrum) in zip(shifts, Spectra['spectra'].items()):
        shifted_intensities = np.interp(spectrum.index - shift, spectrum.index, spectrum.values)
        spectrum[:] = shifted_intensities
        aligned_spectra.append(spectrum)

    Spectra['spectra'] = pd.concat(aligned_spectra, axis=1)
    Spectra['shifts'] = pd.Series(shifts)

def sum_spectra():
    """Add all the spectra together after calibration
    The result is stored in Spectra['total_sum']

    meta data describing shifts is applied here
    """
    global Spectra
    Spectra['total_sum'] = Spectra['spectra'].sum(axis=1)

    if len(Spectra['shifts']) == 0:
        Spectra['shifts'] = pd.Series(np.zeros(len(Spectra['spectra'])))

    meta = '# Shifts applied \n'
    for name, shift in zip(Spectra['spectra'].keys(), Spectra['shifts']):
        meta += '# {}\t {} \n'.format(name, shift)
    meta += '\n'

    Spectra['meta'] = meta

def plot_sum(ax3):
    """Plot the summed spectra on ax3
    """
    plt.sca(ax3)
    sum_line = Spectra['total_sum'].plot()
    plt.xlabel('Pixel / Energy')
    plt.ylabel('Photons')

def calibrate_spectra(zero_E, energy_per_pixel):
    """Convert spectrum x-axis from pixels to energy

    Argument:
    zero_E -- the pixel corresponding to zero energy loss
    energy_per_pixel -- number of meV or eV in one pixel
    """
    global Spectra
    """Calibrate all spectra"""
    Spectra['total_sum'].index = (Spectra['total_sum'].index-zero_E)*energy_per_pixel

def save_spectrum(savefilepath):
    """Save final spectrum

    Argument:
    savefilepath -- path to save file at
    """
    f = open(savefilepath, 'w')
    f.write(Spectra['meta'])
    f.write(Spectra['total_sum'].to_string())
    f.close()

def run_test(search_path='test_data/*.txt'):
    """Calls all functions from the command line
    in order to perform a dummy analysis on test_data/*.txt

    Plot axes contains:
    ax1 -- All spectra
    ax2 -- Shifts applied during lineup
    ax3 -- Summed spectrum before and after calibrating
    which are returned to edit the plots
    """

    # Create plot windows
    plt.figure(1)
    ax1 = plt.subplot(111)
    plt.figure(2)
    ax2 = plt.subplot(111)
    plt.figure(3)
    ax3 = plt.subplot(111)

    # Load spectra
    load_spectra(search_path, get_all_spectra_names(search_path))
    if len(Spectra['spectra']) == 0:
        print("Making fake spectra")
        Spectra['spectra'] = make_fake_spectra()

    # Align spectra
    align_min = 10
    align_max = 70
    plot_spectra(ax1, align_min, align_max)
    first_spectrum_name = Spectra['spectra'].keys()[0]
    shifts = get_shifts_w_mean(first_spectrum_name, align_min, align_max, background=0.5)
    apply_shifts(shifts)
    plot_spectra(ax1, align_min, align_max)

    # Plot the shifts applied
    plot_shifts(ax2)

    # Sum, calibrate to energy and plot
    sum_spectra()
    zero_energy = 40
    energy_per_pixel = 2
    calibrate_spectra(zero_energy, energy_per_pixel)
    plot_sum(ax3)

    # Save result
    save_spectrum('out.dat')

    return ax1, ax2, ax3

if __name__ == "__main__":
    print('Run a test of the code')
    run_test()
