import numpy as np
import pandas as pd

import os, glob

from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt

Experiment = OrderedDict({
'spectra' : pd.DataFrame([]),
'total_sum' : pd.Series([]),
'shifts' : pd.Series([]),
'meta' : ''
})

def load_spectra(search_path, selected_file_names):
    """Load all spectra
    One pandas series will be created per spectrum and stored as
    pandas dataframe Experiment['specta']
    
    Arguments:
    search_path -- string that defines folder containing files
    selected_file_names -- list of filenames to load
    """
    global Experiment
    
    folder = os.path.dirname(search_path)
    
    spectra = []
    for name in selected_file_names:
        data = np.loadtxt(os.path.join(folder, name))
        spectrum = pd.Series(data[:,2], index=np.arange(len(data[:,2])), name=name)
        spectra.append(spectrum)
    
    if spectra != {}:
        Experiment['spectra'] = pd.concat(spectra, axis=1)
    else:
        Experiment['spectra'] = pd.DataFrame([])
        print("No spectra loaded")
    
def plot_spectra(ax1, align_min, align_max):
    """Plot spectra on ax1
    
    Arguments:
    ax1 -- axis to plot on
    align_min -- vertical line defining minimum alignment value
    align_max -- vertical line defining maximum alignment value
    """
    global Experiment
    
    plt.sca(ax1)
    while ax1.lines != []:
        ax1.lines[0].remove()
    
    num_spectra = Experiment['spectra'].shape[1]
    plot_color = iter(matplotlib.cm.spectral(np.linspace(0,1,num_spectra)))
    for name, spectrum in Experiment['spectra'].iteritems():
        spectrum.plot(color=next(plot_color)) #, scalex=False, scaley=False
    plt.xlabel('Pixel / Energy')
    plt.ylabel('Photons')
    
    plt.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})
    
    plt.axvline(x=align_min, color='b')
    plt.axvline(x=align_max, color='b')

def plot_shifts(ax2):
    """Plot the shift values onto ax2"""
    plt.sca(ax2)
    try:
        shifts_line.remove()
    except:
        pass
    shifts_line = plt.plot(Experiment['shifts'], 'ko-', label='Pixel offsets')
    plt.ylabel('Shift (pixels)')    
    plt.xticks(Experiment['shifts'].index, [name for name in Experiment['spectra'].columns], rotation=30)

    
def correlate(ref, align_min, align_max):
    """Determine the shift in pandas index value required to line up spectra with ref
    
    Arguments:
    align_min, align_max -- define range of data used in correlation
    
    zero_shift = np.argmax(np.correlate(ref, ref, mode='Same'))
    xref = np.arange(len(ref))    
    shifts = []
    for name, spectrum in Experiment['spectra'].iteritems():
        
        choose_range = np.logical_and(spectrum.index>align_min, spectrum.index<align_max)
        spec = spectrum[choose_range].values
        
        cross_corr = np.correlate(ref, spec, mode='Same')
        shift = np.argmax(cross_corr) - zero_shift
        shifts.append(shift)
    
    
def align_spectra(ref_name, align_min, align_max):
    """Align the spectra
    Global Experiment['spectra'] will be modified in order to account for energy drifts
    
    Arguments
    ref_name -- name of spectrum for zero energy shift
    """
    global Experiment
    
    if len(Experiment['spectra']) != 0:
        ref_spectrum = Experiment['spectra'][ref_name]
        choose_range = np.logical_and(ref_spectrum.index>align_min, ref_spectrum.index<align_max)
        ref = ref_spectrum[choose_range].values
        correlate(ref, align_min, align_max)
        
        # apply shift
        for shift, (name, spectrum) in zip(Experiment['shifts'], Experiment['spectra'].items()):
            #print("name {} shift {}".format(name, shift))
            shifted_intensities = np.interp(spectrum.index - shift, spectrum.index, spectrum.values)
            spectrum[:] = shifted_intensities
            #spectrum = pd.Series(data[:,2], index=np.arange(len(data[:,2])), name=name)
            aligned_spectra.append(spectrum)
        
        Experiment['spectra'] = pd.concat(aligned_spectra, axis=1)
        #Experiment['spectra'] = pd.DataFrame(aligned_spectra)
        
    else:
        print("Nothing to align")
        return

def sum_spectra():
    """Add all the spectra together after calibration
    The result is stored in Experiment['total_sum']
    
    meta data describing shifts is applied here
    """
    global Experiment
    Experiment['total_sum'] = Experiment['spectra'].sum(axis=1)
    
    if len(Experiment['shifts']) == 0:
        Experiment['shifts'] = pd.Series(np.zeros(len(Experiment['spectra'])))
    
    meta = '# Shifts applied \n'
    for name, shift in zip(Experiment['spectra'].keys(), Experiment['shifts']):
        meta += '# {}\t {} \n'.format(name, shift)
    meta += '\n'
    
    Experiment['meta'] = meta
    
    """Plot the summed spectra on ax3
    """
    plt.sca(ax3)
    sum_line = Experiment['total_sum'].plot()
    plt.xlabel('Pixel / Energy')
    plt.ylabel('Photons')

def calibrate_spectra(zero_E, energy_per_pixel):
    """Convert spectrum x-axis from pixels to energy
    
    Argument:
    zero_E -- the pixel corresponding to zero energy loss
    energy_per_pixel -- number of meV or eV in one pixel
    """
    global Experiment
    """Calibrate all spectra"""
    Experiment['total_sum'].index = (Experiment['total_sum'].index-zero_E)*energy_per_pixel

def save_spectrum(savefilepath):
    """Save final spectrum
    
    Argument:
    savefilepath -- path to save file at
    """
    f = open(savefilepath, 'w')
    f.write(Experiment['meta'])
    f.write(Experiment['total_sum'].to_string())
    f.close()

def run_test():
    """Calls all functions from the command line
    in order to perform a dummy analysis on test_data/*.txt
    
    Plot axes contain:
    ax1 -- All spectra
    ax2 -- Shifts applied during lineup
    ax3 -- Summed spectrum before and after calibrating
    """
    
    # Create plot windows
    plt.figure(1)
    ax1 = plt.subplot(111)
    plt.figure(2)
    ax2 = plt.subplot(111)
    plt.figure(3)
    ax3 = plt.subplot(111)

    # Load spectra
    search_path = 'test_data/*.txt'
    chosen_file_names = [path.split('/')[-1] for path in glob.glob(search_path)]
    load_spectra(search_path, chosen_file_names)

    # Align spectra
    align_min = 10
    align_max = 70
    align_spectra(chosen_file_names[0], align_min, align_max)

    # Plot the shifts applied

    # Sum spectra and display sum
    sum_spectra()

    # Calibrate to energy
    zero_energy = 49
    energy_per_pixel = 2
    calibrate_spectra(zero_energy, energy_per_pixel)
    
    # Save result
    save_spectrum('out.dat')