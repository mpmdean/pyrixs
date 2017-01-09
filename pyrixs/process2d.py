""" Functions for processing 2D Image data. This uses

Typical command line workflow is executed in run_rest()

Parameters
----------
selected_image_name : string
    string specifying image right now a filename
photon_events : array
    two column x, y photon location coordinates
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

import lmfit, os, glob

from pyrixs import loaddata

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

def load_image(search_path, selected_image_name):
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
        two column x, y photon location coordinates
    """
    filename = os.path.join(os.path.dirname(search_path), selected_image_name)
    photon_events = loaddata.getimage(filename)
    return photon_events

def make_fake_image():
    """Return a fake list of photon events.

    Returns
    ----------
    photon_events : array
        two column x, y photon location coordinates
    """
    randomy = 2**11*np.random.rand(1000000)
    choose = (np.exp( -(randomy-1000)**2 / 5 ) +.002) >np.random.rand(len(randomy))
    yvalues = randomy[choose]
    xvalues = 2**11 * np.random.rand(len(yvalues))
    return np.vstack((xvalues, yvalues-xvalues*.02)).transpose()

def plot_image(ax1, photon_events, alpha=0.5, s=1):
    """Plot the image composed of an event list

    Parameters
    ----------
    ax1 : matplotlib axes object
        axes for plotting on
    photon_events : array
        two column x, y photon location coordinates
    alpha : float
        transparency of plotted points (default 0.5)
    s : float
        size of points (default 1)

    Returns
    ----------
    image_artist : matplotlib artist object
        artist from image scatter plot
    """
    plt.sca(ax1)
    ax1.set_axis_bgcolor('black')
    image_artist = plt.scatter(photon_events[:,0], photon_events[:,1], c='white',
            edgecolors='white', alpha=alpha, s=s)
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
    result = GaussianModel.fit(y, x=x, params=params)

    if result.success:
        resolution = [result.best_values[arg] for arg in ['FWHM', 'center', 'amplitude', 'offset']]
        return resolution

def bin_edges_centers(minvalue, maxvalue, binsize):
    """Make bin edges and centers for use in histogram
    The rounding of the bins edges is such that all bins are fully populated.

    Parameters
    -----------
    minvalue/maxvalue : array/array
        minimn/ maximum
    binsize : float (usuallly a whole number)
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
        two column x, y photon location coordinates
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
    x_edges, x_centers = bin_edges_centers(np.nanmin(x), np.nanmax(x), binx)
    y_edges, y_centers = bin_edges_centers(np.nanmin(x), np.nanmax(y), biny)

    H, _, _ = np.histogram2d(x,y, bins=(x_edges, y_edges))

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
    return plt.plot(x, y, 'r-', hold=True)

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
    corrected_y = y - poly(x, curvature[0], curvature[1], 0.)
    pix_edges, pix_centers = bin_edges_centers(np.nanmin(corrected_y),
                                            np.nanmax(corrected_y), biny)
    I, _ = np.histogram(corrected_y, bins=pix_edges)
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
    if xmin == None:
        xmin = np.nanmin(spectrum[:,0])
    if xmax == None:
        xmax = np.nanmax(spectrum[:,0])
    x = np.linspace(xmin, xmax, 10000)
    y = gaussian(x, *resolution)
    return plt.plot(x, y, 'r-', hold=True)

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
        photon_events = load_image(search_path, selected_image_name)
    except (KeyError, IndexError) as e:
        print("No data found: Simulate an image")
        selected_image_name = '<<<SIMULATED>>>'
        photon_events = make_fake_image()
    curvature = fit_curvature(photon_events)
    print("Curvature is {} x^2 + {} x + {}".format(*curvature))

    plt.figure()
    ax1 = plt.subplot(111)
    plt.title('Image {}'.format(selected_image_name))
    plt.figure()
    ax2 = plt.subplot(111)
    plt.title('Spectrum {}'.format(selected_image_name))

    image_artist = plot_image(ax1, photon_events)
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
