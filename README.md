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

--------
Installation guide

1. Creation a python environment to work in (optional)
Download anaonda 
https://www.continuum.io/downloads
conda create --name analysis
source activate analysis
pip install lmfit

2. Install package
https://github.com/mpmdean/pyrixs
Clone or download
Extract
cd pyrixs-master
python setup.py install

3. Open notebook
jpuyter notebook
notebooks > Images_GUI
or
notebooks > Spectra_GUI
