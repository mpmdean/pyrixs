""" Functions for processing 2D Image data. This uses

Typical command line workflow is executed in run_rest()

Parameters
----------
selected_image_name : string
    string specifying image right now a filename
photon_events : array
    three column x, y, I photon locations and intensities
curvature : array
    n2d order polynominal defining image curvature
    np.array([x^2 coef, x coef, offset])
image_meta : string
    container for metadata
spectrum: array
    binned spectrum two column defining
    pixel, intensity
resolution_values : array
    parameters defining gaussian from fit_resolution
"""

import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import reduce

import lmfit, os, glob

from pyrixs import loaddata

from scipy.stats import binned_statistic

CONSTANT_OFFSET = 500

def get_all_image_names(search_path):
    """Returns list of image names meeting the folder search term.

    Parameters
    ----------
    search_path : string
        defines folder containing images

    Returns
    ----------
    image_names : list
        string showing filenames
    """
    paths = glob.glob(search_path)
    return [path.split('/')[-1] for path in paths]

def load_photon_events(search_path, selected_image_name):
    """Return image based on file location

    Parameters
    ----------
    search_path : string
        defines folder containing images
    selected_image_name : list
        image filename to load

    Returns
    ----------
    photon_events : array
        three column x, y, I photon locations and intensities
    """
    filename = os.path.join(os.path.dirname(search_path), selected_image_name)
    photon_events = loaddata.get_image(filename)
    return photon_events

def make_fake_image():
    """Return a fake list of photon events.

    Returns
    ----------
    photon_events : array
        three column x, y, I photon locations
    """
    randomy = 2**11*np.random.rand(1000000)
    choose = (np.exp( -(randomy-1000)**2 / 5 ) +.002) >np.random.rand(len(randomy))
    yvalues = randomy[choose]
    xvalues = 2**11 * np.random.rand(len(yvalues))
    I = np.ones(xvalues.shape)
    return np.vstack((xvalues, yvalues-xvalues*.02, I)).transpose()

def plot_scatter(ax1, photon_events, **kwargs):
    """Make a scatterplot

    Parameters
    ----------
    ax1 : matplotlib axes object
        axes for plotting on
    photon_events : array
        two column x, y, I photon locations and intensities
    **kwargs : dictionary
        passed onto matplotlib.pyplot.scatter. Add pointsize to multiply all
        point sizes by a fixed factor.
    Returns
    ----------
    image_artist : matplotlib artist object
        artist from image scatter plot
    """
    plt.sca(ax1)
    ax1.set_axis_bgcolor('black')

    defaults = {'c': 'white', 'edgecolors' : 'white', 'alpha' : 0.5,
                'pointsize' : 1}

    kwargs.update({key:val for key, val in defaults.items() if key not in kwargs})

    pointsize = kwargs.pop('pointsize')
    image_artist = plt.scatter(photon_events[:,0], photon_events[:,1],
                s=photon_events[:,2]*pointsize, **kwargs)
    return image_artist

def plot_imshow(ax1, photon_events, **kwargs):
    """Make an image of the photon_events

    Parameters
    ----------
    ax1 : matplotlib axes object
        axes for plotting on
    photon_events : array
        two column x, y, I photon locations and intensities
    **kwargs : dictionary
        passed onto matplotlib.pyplot.imshow.
    Returns
    ----------
    image_artist : matplotlib artist object
        artist from image scatter plot
    """
    plt.sca(ax1)

    image = loaddata.photon_events_to_image(photon_events)
    defaults = {'interpolation' : 'None', 'cmap' : 'gray',
         'vmin' : np.percentile(image,1), 'vmax' : np.percentile(image,99)}

    kwargs.update({key:val for key, val in defaults.items() if key not in kwargs})

    image_artist = ax1.imshow(image, **kwargs)
    plt.gcf().colorbar(image_artist)
    ax1.axis('tight')
    ax1.invert_yaxis()

    return image_artist

def poly(x, p2, p1, p0):
    """Third order polynominal function for fitting curvature.
    Returns p2*x**2 + p1*x + p0
    """
    return p2*x**2 + p1*x + p0

def gaussian(x, FWHM=3., center=100., amplitude=10., offset=0.):
    """Gaussian function defined by full width at half maximum FWHM
    with an offset for fitting resolution.
    """
    two_sig_sq = (FWHM / (2 * np.log(2) ) )**2
    return amplitude * np.exp( -(x-center)**2/two_sig_sq ) + offset

def fit_poly(x_centers, offsets):
    """Fit curvature to vaues for curvature offsets.

    Parameters
    ----------
    x_centers, y_centers : float, float
        Shifts of the isoenergetic line as a function of column, x

    Returns
    --------
    result : lmfit result object
        object describing polynominal fit
    """
    poly_model = lmfit.Model(poly)
    params = poly_model.make_params()
    params['p0'].value = offsets[0]
    params['p1'].value = 0.
    params['p2'].value = 0.
    result = poly_model.fit(offsets, x=x_centers, params=params)
    if not result.success:
        print("Fitting failed")
    return result


def fit_resolution(spectrum, xmin=-np.inf, xmax=np.inf):
    """Fit a Gaussian model to ['spectrum'] in order to determine resolution_values.

    Parameters
    ----------
    spectrum: array
        binned spectrum two column defining
        pixel, intensity
    xmin/xmax : float
        minimum/maximum value for fitting range

    Returns
    ----------
    resolution : array
        values parameterizing gaussian function
        ['FWHM', 'center', 'amplitude', 'offset']
    """
    allx = spectrum[:,0]
    choose = np.logical_and(allx>xmin, allx<=xmax)
    x = allx[choose]
    y = spectrum[:,1][choose]

    GaussianModel = lmfit.Model(gaussian)
    params = GaussianModel.make_params()
    params['center'].value = x[np.argmax(y)]
    params['FWHM'].set(min = 0)
    result = GaussianModel.fit(y, x=x, params=params)

    if result.success:
        resolution = [result.best_values[arg] for arg in ['FWHM', 'center', 'amplitude', 'offset']]
        return resolution
    else:
        print('Resolution fit failed! Try changing initial parameters!')
        return None

def bin_edges_centers(minvalue, maxvalue, binsize):
    """Make bin edges and centers for use in histogram
    The rounding of the bins edges is such that all bins are fully populated.

    Parameters
    -----------
    minvalue/maxvalue : array/array
        minimun/ maximum
    binsize : float (usually a whole number)
        difference between subsequnt points in edges and centers array

    Returns
    -----------
    edges : array
        edge of bins for use in np.histrogram
    centers : array
        central value of each bin. One shorter than edges
    """
    edges = binsize * np.arange(minvalue//binsize + 1, maxvalue//binsize)
    centers = (edges[:-1] + edges[1:])/2
    return edges, centers

def get_curvature_offsets(photon_events, binx=64, biny=1):
    """ Determine the offests that define the isoenergetic line.
    This is determined as the maximum of the cross correlation function with
    a reference taken from the center of the image.

    Parameters
    ------------
    photon_events : array
        three column x, y, I photon locations and intensities
    binx/biny : float/float (usually whole numbers)
        width of columns/rows binned together prior to computing
        convolution. binx should be increased for noisy data.

    Returns
    -------------
    x_centers : array
        columns positions where offsets were determined
        i.e. binx/2, 3*binx/2, 5*binx/2, ...
    offests : array
        np.array of row offsets defining curvature. This is referenced
        to the center of the image.
    """
    x = photon_events[:,0]
    y = photon_events[:,1]
    I = photon_events[:,2]
    x_edges, x_centers = bin_edges_centers(np.nanmin(x), np.nanmax(x), binx)
    y_edges, y_centers = bin_edges_centers(np.nanmin(y), np.nanmax(y), biny)

    H, _, _ = np.histogram2d(x,y, bins=(x_edges, y_edges), weights=I)
    
    ref_column = H[H.shape[0]//2, :]

    offsets = np.array([])
    for column in H:
        cross_correlation = np.correlate(column, ref_column, mode='same')
        offsets = np.append(offsets, y_centers[np.argmax(cross_correlation)])
    return x_centers, offsets - offsets[offsets.shape[0]//2]

def fit_curvature(photon_events, binx=32, biny=1, CONSTANT_OFFSET=500):
    """Get offsets, fit them and return polynomial that defines the curvature

    Parameters
    -------------
    photon_events : array
        two column x, y photon location coordinates
    binx/biny : float/float (usually whole numbers)
        width of columns/rows binned together prior to computing
        convolution. binx should be increased for noisy data.
    CONSTANT_OFFSET : float
        offset is pass into last value of curvature

    Returns
    -----------

    """
    x_centers, offsets = get_curvature_offsets(photon_events, binx=binx, biny=biny)
    
    result = fit_poly(x_centers, offsets)
    curvature = np.array([result.best_values['p2'], result.best_values['p1'],
                            CONSTANT_OFFSET])
    return curvature

def plot_curvature(ax1, curvature, photon_events):
    """ Plot a red line defining curvature on ax1

    Parameters
    ----------
    ax1 : matplotlib axes object
        axes for plotting on
    curvature : array
        n2d order polynominal defining image curvature
        np.array([x^2 coef, x coef, offset])
    photon_events : array
        two column x, y photon location coordinates

    Returns
    ---------
    curvature_artist : matplotlib artist object
        artist from image scatter plot
    """
    x = np.arange(np.nanmax(photon_events[:,0]))
    y = poly(x, *curvature)
    #return plt.plot(x, y, 'r-', hold=True)
    return plt.plot(x, y, 'r-')

def extract(photon_events, curvature, biny=1.):
    """Apply curvature to photon events to create pixel versus intensity spectrum

    Parameters
    ----------
    photon_events : array
        two column x, y photon location coordinates
    curvature : array
        n2d order polynominal defining image curvature
        np.array([x^2 coef, x coef, offset])
    biny : float (usuallly a whole number)
        difference between subsequnt points spectrum
    """
    x = photon_events[:,0]
    y = photon_events[:,1]
    I = photon_events[:,2]
    corrected_y = y - poly(x, curvature[0], curvature[1], 0.)
    pix_edges, pix_centers = bin_edges_centers(np.nanmin(corrected_y),
                                            np.nanmax(corrected_y), biny)
    I, _ = np.histogram(corrected_y, bins=pix_edges, weights=I)
    spectrum = np.vstack((pix_centers, I)).transpose()
    return spectrum

def plot_resolution(ax2, spectrum):
    """ Plot blue points defining the spectrum on ax2

    Parameters
    ------------
    ax2 : matplotlib axes object
    spectrum: array
        binned spectrum two column defining
        pixel, intensity

    Returns
    -----------
    spectrum_artist : matplotlib artist
        Resolution plotting object
    """
    plt.sca(ax2)
    spectrum_artist = plt.plot(spectrum[:,0], spectrum[:,1], 'b.')
    plt.xlabel('pixels')
    plt.ylabel('Photons')
    return spectrum_artist

def plot_resolution_fit(ax2, spectrum, resolution, xmin=None, xmax=None):
    """Plot the gaussian fit to the resolution function

    Parameters
    -----------
    ax2 : matplotlib axes object
        axes to plot on
    spectrum: array
        binned spectrum two column defining
        pixel, intensity
    xmin/xmax : float/float
        range of x values to plot over (the same as fitting range)
    resolution_values : array
        parameters defining gaussian from fit_resolution
    """
    plt.sca(ax2)
    if xmin is None:
        xmin = np.nanmin(spectrum[:,0])
    if xmax is None:
        xmax = np.nanmax(spectrum[:,0])
    x = np.linspace(xmin, xmax, 10000)
    y = gaussian(x, *resolution)
    #return plt.plot(x, y, 'r-', hold=True)
    return plt.plot(x, y, 'r-')

def clean_image_threshold(photon_events, thHigh):
    """ Remove cosmic rays and glitches using a fixed threshold count.

    Parameters
    ------------
    photon_events : array
        three column x, y, z with location coordinates (x,y) and intensity (z)
    thHigh: float
        Threshold limit. Pixels with counts above/below +/- thHigh will be removed from image.

    Returns
    -----------
    clean_photon_events : array
        Cleaned photon_events
    changed_pixels: float
        1 - ratio between of removed and total pixels.
    """
    bad_indices = np.logical_and(photon_events[:,2] < thHigh, photon_events[:,2] > -1.*thHigh)
    clean_photon_events = photon_events[bad_indices,:]
    changed_pixels = 1.0 - clean_photon_events.shape[0] / photon_events.shape[0]*1.0
    return clean_photon_events, changed_pixels


def clean_image_std(photon_events, sigma, curvature):
    """ Remove cosmic rays and glitches based on the stardard deviation for each isoenergetic row. 
    Values beyond +-sigma[i]*std are removed. Note that it is usefull to have sigma.size >= 3 in
    decreasing order. This is because in a single iteraction, a big glitch will affect the mean 
    and std, potentially masking a small one.

    Parameters
    ------------
    photon_events : array
        three column x, y, z with location coordinates (x,y) and intensity (z)
    sigma: list or array
        factor of standard deviation that is used for threshold,
        i.e. values beyond +-sigma[i]*std are removed. The sigma[i]<=0 are ignored.
    curvature : array
        n2d order polynominal defining image curvature
        np.array([x^2 coef, x coef, offset])
    
    Returns
    -----------
    clean_photon_events : array
        cleaned photon_events.
    changed_pixels: float
        ratio between of changed and total pixels.
    """
    
    #Sigmas <= 0 are ignored.
    sigma = np.array(sigma)
    sigma = sigma[sigma > 0]

    #Remove curvature and convert photon_events into image
    x = photon_events[:,0]
    y = photon_events[:,1]
    I = photon_events[:,2]
    
    ## I'm doing the edges/centers here to preserve array size.
    corrected_y = y - poly(x, curvature[0], curvature[1], 0.)
    y_edges = 1.0 * np.arange(np.nanmin(corrected_y)//1.0, np.nanmax(corrected_y)//1.0 + 2)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    
    cleanI = np.copy(I)    
    for sig in sigma:
        #Creating index to convert the mean and std binned arrays of y_centers.size into cleanI.size
        index = np.digitize(corrected_y,y_edges)
        
        mean,_,_ = binned_statistic(corrected_y,cleanI,statistic='mean',bins=y_edges)
        std,_,_ = binned_statistic(corrected_y,cleanI,statistic='std',bins=y_edges)
        mean_array = mean[index-1]
        std_array = std[index-1]

        bad_indices = np.logical_or(cleanI < (mean_array - sig*std_array), cleanI > (mean_array + sig*std_array))
        
        cleanI = cleanI[np.logical_not(bad_indices)]
        corrected_y = corrected_y[np.logical_not(bad_indices)]
        x = x[np.logical_not(bad_indices)]
        y = y[np.logical_not(bad_indices)]

    clean_photon_events = np.vstack((x,y,cleanI)).transpose()
    changed_pixels = 1. - cleanI.size/I.size
    
    return clean_photon_events, changed_pixels

def gen_dark(photon_events, start_row_index=100, end_row_index=300):
    """ Generate image for background subtraction without real dark image.
    Data is taken between row start_row_index and end_row_index the 50%
    percentile is taken to minimize sensitivity to spikes
           
    Parameters
    ------------
    photon_events : array
        three column x, y, z with location coordinates (x,y) and intensity (z)
    start_row_index: int
        first row of the background
    end_row_index : int
        last row of the background
    
    Returns
    -----------
    dark_photon_events : array
        generated dark image
    """
    
    index = np.logical_and(photon_events[:,1] > start_row_index, photon_events[:,1] < end_row_index)
    
    dark_photon_events = np.copy(photon_events)
    dark_photon_events[:,2] = np.percentile(photon_events[index,2], 50)
    return dark_photon_events

def run_test(search_path='../test_images/*.h5'):
    """Run at test of the code.
    This can also be used as a reference for command-line analysis.

    Parameters
    -------------
    search_path : string
        string to fine images passed to get_all_image_names

    Returns
    -------------
    ax1 : matplotlib axis object
        axes used to plot the image
    ax2 : matplotlib axis object
        axes used to plot the spectrum
    """
    try:
        selected_image_name = get_all_image_names(search_path)[0]
        photon_events = load_photon_events(search_path, selected_image_name)
    except (KeyError, IndexError) as e:
        print("No data found: Simulate an image")
        selected_image_name = '<<<SIMULATED>>>'
        photon_events = make_fake_image()
    curvature = fit_curvature(photon_events)
    print("Curvature is {} x^2 + {} x + {}".format(*curvature))

    fig1, ax1 = plt.subplots()
    plt.title('Image {}'.format(selected_image_name))
    fig2, ax2 = plt.subplots()
    plt.title('Spectrum {}'.format(selected_image_name))

    image_artist = plot_scatter(ax1, photon_events)
    spectrum = extract(photon_events, curvature)
    plot_curvature(ax1, curvature, photon_events)
    resolution = fit_resolution(spectrum)
    plot_resolution(ax2, spectrum)
    plot_resolution_fit(ax2, spectrum, resolution)

    print("Resolution is {}".format(resolution[0]))
    return ax1, ax2

if __name__ == "__main__":
    """Run test of code"""
    print('Run a test of the code')
    run_test()
