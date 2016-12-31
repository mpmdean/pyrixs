import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
from collections import OrderedDict

import lmfit, os, glob

from pyrixs import loaddata

#############################
# function for processing 2D image data
# everything is stored in ordered dictionary Image
#############################

Image = OrderedDict({
'photon_events' : np.array([]),
'name' : '',
'curvature' : np.array([0, 0, 500]),
'image_meta' : '',
'spectrum' : np.array([])
})

global Image

def get_all_image_names(search_path):
    """Returns list of image names meeting the folder search term."""
    paths = glob.glob(search_path)
    return [path.split('/')[-1] for path in paths]

def load_image(search_path, selected_image_name):
    """Read image into Image['photon_events']

    Arguments:
    search_path -- string that defines folder containing images
    selected_image_name -- list of image filenames to load
    """
    global Image
    filename = os.path.join(os.path.dirname(search_path), selected_image_name)
    Image['photon_events'] = loaddata.getimage(filename)
    Image['name'] = selected_image_name

def make_fake_image():
    """Return a fake list of photon events."""
    randomy = 2**11*np.random.rand(1000000)
    choose = (np.exp( -(randomy-1000)**2 / 5 ) +.002) >np.random.rand(len(randomy))
    yvalues = randomy[choose]
    xvalues = 2**11 * np.random.rand(len(yvalues))
    return np.vstack((xvalues, yvalues-xvalues*.02)).transpose()

def plot_image(ax1, alpha=0.5, s=1):
    """Plot the image composed of an event list

    Arguments:
    ax1 -- axes for plotting on
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
    """Third order polynominal function for fitting curvature that
    returns p2*x**2 + p1*x + p0 .
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

def fit_resolution(xmin=-np.inf, xmax=np.inf):
    """Fit a Gaussian model to Image['spectrum'] in order to determine resolution_values.

    Arguments:
    xmin, xmax -- define range data to be used
    """
    allx = Image['spectrum'][:,0]
    choose = np.logical_and(allx>xmin, allx<=xmax)
    x = allx[choose]
    y = Image['spectrum'][:,1][choose]

    GaussianModel = lmfit.Model(gaussian)
    params = GaussianModel.make_params()
    params['center'].value = x[np.argmax(y)]
    result = GaussianModel.fit(y, x=x, params=params)

    if result.success:
        resolution_values = [result.best_values[arg] for arg in ['FWHM', 'center', 'amplitude', 'offset']]
        return resolution_values


def get_curvature_offsets(binx=64, biny=1):
    """ Determine the offests that define the isoenergetic line.
    This is determined as the maximum of the cross correlation function with
    a reference taken from the center of the image.

    Arguments:
    binx/biny -- width of columns/rows binned together prior to computing
                 convolution. binx should be increased for noisy data.
    """
    global Image
    x = Image['photon_events'][:,0]
    y = Image['photon_events'][:,1]
    x_edges = binx * np.arange(np.floor(np.nanmax(x)/binx))
    x_centers = (x_edges[:-1] + x_edges[1:])/2
    y_edges = biny * np.arange(np.floor(np.nanmax(y)/biny))
    y_centers = (y_edges[:-1] + y_edges[1:])/2

    cenpix = np.nanmax(x)/2
    yvals = y[np.logical_and(x>(cenpix-binx/2), x<=(cenpix+binx/2))]
    ref, _ = np.histogram(yvals, bins=y_edges)

    offsets = []
    for x_min, x_max in zip(x_edges[:-1], x_edges[1:]): # possible with 2D hist?
        yvals = y[np.logical_and(x>x_min, x<=x_max)]
        I, _ = np.histogram(yvals, bins=y_edges)
        ycorr = np.correlate(I, ref, mode='same')
        offsets.append(y_centers[np.argmax(ycorr)])

    return x_centers, np.array(offsets)

def fit_curvature(binx=64, biny=1):
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

    Arguments:
    biny -- width of rows binned together in histogram"""
    global Image
    XY = Image['photon_events']
    curvature_corrected_y = XY[:,1] - poly(XY[:,0],
                                           Image['curvature'][0],
                                           Image['curvature'][1],
                                           0.)
    maxy = np.nanmax(curvature_corrected_y)
    pix_edges = biny/2. + biny * np.arange(np.floor(maxy/biny))
    I, _ = np.histogram(curvature_corrected_y, bins=pix_edges)
    pix_centers = (pix_edges[0:-1] + pix_edges[1:]) / 2
    spectrum = np.vstack((pix_centers, I)).transpose()
    Image['spectrum'] = spectrum
    return spectrum

def plot_resolution(ax2):
    """ Plot blue points defining the spectrum on ax2"""
    plt.sca(ax2)
    spectrum = Image['spectrum']
    spectrum_line = plt.plot(spectrum[:,0], spectrum[:,1], 'b.')
    plt.xlabel('pixels')
    plt.ylabel('Photons')

def plot_resolution_fit(ax2, resolution_values, xmin=None, xmax=None):
    """Plot the gaussian fit to the resolution function

    Arguments:
    ax2 -- axes to plot on
    xmin/xmax -- range of x values to plot over (same as fitting range)
    resolution_values -- parameters defining gaussian from fit_resolution
    """
    plt.sca(ax2)
    if xmin == None:
        xmin = np.nanmin(Image['spectrum'][:,0])
    if xmax == None:
        xmax = np.nanmax(Image['spectrum'][:,0])
    x = np.linspace(xmin, xmax, 10000)
    y = gaussian(x, *resolution_values)
    resolution_line = plt.plot(x, y, 'r-', hold=True)

def run_test(search_path='../test_images/*.h5'):
    """Run at test of the code.
    This can also be used as a reference for command-line analysis.

    Arguments:
    search_path -- string to fine images passed to get_all_image_names
    """
    try:
        selected_image_name = get_all_image_names(search_path)[0]
        load_image(search_path, selected_image_name)
    except (KeyError, IndexError) as e:
        Image['photon_events'] = make_fake_image()
        Image['name'] = 'fake data'
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
    resolution_values = fit_resolution()
    plot_resolution(ax2)
    plot_resolution_fit(ax2, resolution_values)

    print("Resolution is {}".format(resolution_values[0]))

if __name__ == "__main__":
    """Run test of code"""
    print('Run a test of the code')
    run_test()
