from setuptools import setup
import versioneer

setup(name='pyrixs',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Python based analysis for RIXS images and spectra',
      url='https://github.com/mpmdean/pyrixs',
      author='Mark P. M. Dean',
      author_email='mdean@bnl.gov',
      license='MIT',
      packages=['pyrixs'],
      install_requires=['lmfit'],
      zip_safe=False)
