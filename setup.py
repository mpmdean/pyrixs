from setuptools import setup

setup(name='pyrixs',
      version='0.1',
      description='Python based analysis for RIXS images and spectra',
      url='https://github.com/mpmdean/pyrixs',
      author='Mark P. M. Dean',
      author_email='mdean@bnl.gov',
      license='MIT',
      packages=['pyrixs'],
      install_requires=['lmfit'],
      zip_safe=False)
