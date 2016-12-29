import numpy as np
import pandas as pd

import lmfit, h5py, os, glob

from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt

#############################
# GENERAL FUNCTIONS
#############################

def get_all_file_names(search_path):
    """Returns list of image names meeting the folder search term"""
    paths = glob.glob(search_path)
    return [path.split('/')[-1] for path in paths]


#############################
# FUNCTIONS FOR IMAGES
#############################

global Image

Image = OrderedDict({
'photon_events' : np.array([]),
'name' : '',
'curvature' : np.array([0, 0, 500]),
'image_meta' : '',
'spectrum' : np.array([])
})



def load_image(search_path, selected_image_name):
    """Read image into Image['photon_events']
    
    Arguments:
    search_path -- string that defines folder containing images
    selected_image_name -- list of image filenames to load
    """
    global Image
    h5file = h5py.File(os.path.join(os.path.dirname(search_path), selected_image_name))
    Image['photon_events'] = h5file['entry']['analysis']['events'].value
    Image['name'] = selected_image_name

def plot_image(ax1, alpha=0.5, s=1):
    """Plot the image composed of an event list
    
    Arguments:
    alpha -- transparency of plotted points (default 0.5)
    s --  size of points (default 1)
    """
    plt.sca(ax1)
    ax1.set_axis_bgcolor('black')
    photon_events = Image['photon_events']
    plt.scatter(photon_events[:,0], photon_events[:,1], c='white',
            edgecolors='white', alpha=alpha, s=s)
    plt.title(Image['name'])
    plt.xlim([-100, 1700])

def plot_curvature(ax1):
    """ Plot a red line defining curvature on ax1"""
    x = np.arange(np.nanmax(Image['photon_events'][:,0]))
    y = np.polyval(Image['curvature'], x)
    curvature_line = plt.plot(x, y, 'r-', hold=True)

def poly(x, p2, p1, p0):
    """Third order polynominal for fitting curvature
    p2*x**2 + p1*x + p0"""
    return p2*x**2 + p1*x + p0

def gaussian(x, FWHM=3., center=100., amplitude=10., offset=0.):
        """Gaussian defined by full width at half maximum FWHM
        with an offset for fitting resolution
        """
        two_sig_sq = (FWHM / (2 * np.log(2) ) )**2
        return amplitude * np.exp( -(x-center)**2/two_sig_sq ) + offset

def fit_poly(x_centers, offsets):
    """Fit curvature to vaues for curvature offsets
    
    Arguments:
    x_centers, y_centers = Shifts of the isoenergetic line as a function of column, x
    """
    poly_model = lmfit.Model(poly)
    params = poly_model.make_params()
    params['p0'].value = offsets[0]
    for i, arg in enumerate(['p2', 'p1']):
        params[arg].value = Image['curvature'][i]
        
    result = poly_model.fit(offsets, x=x_centers, params=params)
    
    if result.success:
        curvature_values = [result.best_values[arg] for arg in ['p2', 'p1', 'p0']]
        return curvature_values

def fit_resolution(x, y):
    """Gaussian model with offset to x,y to determine resolution"""
    GaussianModel = lmfit.Model(gaussian)
    params = GaussianModel.make_params()
    params['center'].value = x[np.argmax(y)]
    result = GaussianModel.fit(y, x=x, params=params) 
    
    if result.success:
        resolution_values = [result.best_values[arg] for arg in ['FWHM', 'center', 'amplitude', 'offset']]
        return resolution_values

def get_curvature_offsets(binx=16., biny=1.):
    """ Determine the offests that define the isoenergetic line 
    
    Arguments:
    binx/biny -- width of columns/rows binned together prior to computing convolution
    
    """
    global Image
    M = Image['photon_events']
    x = M[:,0]
    y = M[:,1]
    x_edges = binx * np.arange(np.floor(np.nanmax(x)/binx))
    x_centers = (x_edges[:-1] + x_edges[1:])/2
    y_edges = biny * np.arange(np.floor(np.nanmax(y)/biny))
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    
    # compute reference from center of image
    cenpix = np.nanmax(x)/2 + 0.
    yvals = y[np.logical_and(x>(cenpix-binx/2), x<=(cenpix+binx/2))]
    ref, _ = np.histogram(yvals, bins=y_edges)
    
    offsets = []
    for x_min, x_max in zip(x_edges[:-1], x_edges[1:]): # possible with 2D hist?
        yvals = y[np.logical_and(x>x_min, x<=x_max)]
        I, _ = np.histogram(yvals, bins=y_edges)
        ycorr = np.correlate(I, ref, mode='same')
        offsets.append(y_centers[np.argmax(ycorr)])
    
    return x_centers, np.array(offsets)

def fit_curvature(binx=16., biny=1.):
    """Get offsets, fit them with a polynominal and assign values to Image
    
    Arguments:
    binx/biny -- width of columns/rows binned together prior to computing convolution
    """
    global Image
    x_centers, offsets = get_curvature_offsets(binx=binx, biny=biny)
    curvature_values = fit_poly(x_centers, offsets)
    Image['curvature'] = curvature_values
    return curvature_values

def plot_curvature(ax1):
    """ Plot a red line defining curvature on ax1"""
    plt.sca(ax1)
    x = np.arange(np.nanmax(Image['photon_events'][:,0]))
    y = np.polyval(Image['curvature'], x)
    curvature_line = plt.plot(x, y, 'r-', hold=True)

def extract(biny=1.):
    """Apply curvature to photon events to create pixel versus intensity spectrum
    Argument
    biny -- width of rows binned together in histogram"""
    global Image
    XY = Image['photon_events']
    curvature_corrected_y = XY[:,1] - poly(XY[:,0], Image['curvature'][0], Image['curvature'][1], 0.)
    maxy = np.nanmax(curvature_corrected_y)
    #print('maxy = {}'.format(maxy))
    pix_edges = biny/2. + biny * np.arange(np.floor(maxy/biny))
    I, _ = np.histogram(curvature_corrected_y, bins=pix_edges)
    pix_centers = (pix_edges[0:-1] + pix_edges[1:]) / 2
    spectrum = np.vstack((pix_centers, I)).transpose()
    Image['spectrum'] = spectrum
    return spectrum

def plot_resolution(ax2):
    """ Plot a red line from the resolution spectrum on ax2"""
    plt.sca(ax2)
    spectrum = Image['spectrum']
    spectrum_line = plt.plot(spectrum[:,0], spectrum[:,1], 'b.')
    plt.xlabel('pixels')
    plt.ylabel('Photons')

def plot_resolution_fit(ax2, xmin, xmax, resolution_values):
    """Plot the gaussian fit to the resolution function"""
    plt.sca(ax2)
    x = np.linspace(xmin, xmax, 1000)
    y = gaussian(x, *resolution_values)
    resolution_line = plt.plot(x, y, 'r-', hold=True)
    
def test_images():
    """Calls images functions from the command line
    in order to perform a dummy analysis
    
    Plot axes contain:
    ax1 -- Image and curvature
    """
    global Image
    search_path = 'test_images/*.h5'
    selected_image_name = get_all_file_names(search_path)[0]

    load_image(search_path, selected_image_name)
    curvature_values = fit_curvature()
    Image['curvature'][0:2] = curvature_values[0:2]
    print("Curvature is {} x^2 + {} x + {}".format(*curvature_values))

    plt.figure()
    ax1 = plt.subplot(111)
    plt.figure()
    ax2 = plt.subplot(111)

    plot_image(ax1)
    spectrum = extract()
    plot_curvature(ax1)
    plot_resolution(ax2)

    resolution_values = fit_resolution(spectrum[:,0], spectrum[:,1])
    print("Resolution is {}".format(resolution_values[0]))

    plot_resolution_fit(ax2, 0, 1000, resolution_values)

#############################
# FUNCTIONS FOR SPECTRA 
#############################

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
        spectrum.plot(color=next(plot_color))
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
    """
    global Experiment
    
    ref_spectrum = Experiment['spectra'][ref_name]
    choose_range = np.logical_and(ref_spectrum.index>align_min, ref_spectrum.index<align_max)
    ref = ref_spectrum[choose_range].values

    return correlate(ref, Experiment['spectra'], align_min, align_max, background=background)

def get_shifts_w_mean(ref_name, align_min, align_max, background=0.):
    """Get shifts using cross correlation.
    The mean of the spectra after an initial alignment i used as a reference
    
    Arguments
    ref_name -- name of spectrum for zero energy shift
    align_min, align_max -- define range of data used in correlation
    background -- subtract off this value before cross-correlation (default 0)
    """
    global Experiment
    
    # first alignment
    shifts = get_shifts(ref_name, align_min, align_max, background=background)

    # do a first pass alignment
    first_pass_spectra = []
    spectra = Experiment['spectra'].copy()
    for shift, (name, spectrum) in zip(shifts, spectra.items()):
        shifted_intensities = np.interp(spectrum.index - shift, spectrum.index, spectrum.values)
        spectrum[:] = shifted_intensities
        first_pass_spectra.append(spectrum)

        ref_spectrum = pd.concat(first_pass_spectra, axis=1).mean(axis=1)

        choose_range = np.logical_and(ref_spectrum.index>align_min, ref_spectrum.index<align_max)
        ref = ref_spectrum[choose_range].values
        return correlate(ref, Experiment['spectra'], align_min, align_max, background=background)

def apply_shifts(shifts):
    """ Apply shifts values to Experiments['spectra'] and update Experiments['shifts']"""
    global Experiment
    aligned_spectra = []
    for shift, (name, spectrum) in zip(shifts, Experiment['spectra'].items()):
        shifted_intensities = np.interp(spectrum.index - shift, spectrum.index, spectrum.values)
        spectrum[:] = shifted_intensities
        aligned_spectra.append(spectrum)
        
    Experiment['spectra'] = pd.concat(aligned_spectra, axis=1)
    Experiment['shifts'] = pd.Series(shifts)

        
    
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
    
def plot_sum(ax3):
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

def run_RIXS_test():
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
    load_spectra(search_path, get_all_file_names(search_path))

    # Align spectra
    align_min = 10
    align_max = 70
    plot_spectra(ax1, align_min, align_max)
    shifts = get_shifts_w_mean(get_all_file_names(search_path)[0], align_min, align_max, background=0.5)
    apply_shifts(shifts)
    plot_spectra(ax1, align_min, align_max)

    # Plot the shifts applied
    plot_shifts(ax2)

    # Sum spectra and display sum
    sum_spectra()
    plot_sum(ax3)

    # Calibrate to energy
    zero_energy = 49
    energy_per_pixel = 2
    calibrate_spectra(zero_energy, energy_per_pixel)
    plot_sum(ax3)
    
    # Save result
    save_spectrum('out.dat')

if __name__ == "__main__":
    print('Run a test of the code')
    run_RIXS_test()
    