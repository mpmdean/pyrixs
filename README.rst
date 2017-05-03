pyrixs
=========================

Python 3 based analysis for RIXS images and spectra compatible with command line and a GUI based on jupyter widgets.

Installation Instructions
=========================


1. Create a python environment to work in (optional).

    Download and install `anaconda <https://www.continuum.io/downloads>`_.

    Create a conda environment:
    ::
        conda create --name <name_of_enviroment>
    where <name_of_enviroment> is what you want to call the environment. N.B. python 3 is required, which should be the default, but can be explicitly requested by appending ``python=3`` to the ``conda create`` command above.


    Activate the environment:
    ::
        source activate <name_of_enviroment>

2. Install package.

    Download and extract `pyrixs package <https://github.com/mpmdean/pyrixs>`_.

    Change directory and install pyrixs and additional lmfit package:
    ::
        cd pyrixs-master
        python setup.py install
        conda install pandas h5py ipywidgets traitlets pillow
        conda install -c conda-forge lmfit

        Optional:
        conda install nexusformat


    Activate the Javascript widget
    ::
        jupyter nbextension enable --py --sys-prefix widgetsnbextension


3. Launch analysis session.

    Open jupyer:
    ::
        jupyter notebook

    Navigate to *notebooks* and click *Images_GUI* or *Spectra_GUI*.
