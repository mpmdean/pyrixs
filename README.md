# pyrixs
Python based analysis for RIXS images and spectra compatible with command line and a GUI based on jupyter widgets

Likely only works with modern Python 3 due to dependency of ipywidgets

Uses packages:
lmfit
ipywidgets
traitlets

IPython
numpy
scipy
matplotlib


Installation Instructions
=========================

1. Create a python environment to work in (optional).

    Download and install `anaconda <https://www.continuum.io/downloads>`_.

    Create a conda environment:
    ::
        conda create --name *<name_of_enviroment>*
    where *<name_of_enviroment>* is what you want to call the environment.

    Activate the environment:
    ::
        source activate *<name_of_enviroment>*

2. Install package.

    Download and extract `pyrixs package <https://github.com/mpmdean/pyrixs>`_.

    Change directory and install pyrixs and additional lmfit package:
    ::
        cd pyrixs-master
        python setup.py install
        pip install lmfit ipywidgets traitlets

3. Launch analysis session.

    Open jupyer:
    ::
        jupyter notebook

    Navigate to *notebooks* and click *Images_GUI* or *Spectra_GUI*.
