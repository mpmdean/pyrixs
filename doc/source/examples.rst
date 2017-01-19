Use of process 1D
========

The module ``pyrixs.process1d`` deals with aligning and calibrating 1d spectra. Data are stored in Pandas dataframe ``spectra``. Typical workflow involves determining the ``shifts`` required to align all the ``spectra``, summing them to one ``spectrum'' and then calibrating the energy loss x-axis.

The notebooks provide the easiest most straightforward access to these functions, but the ``pyrixs.process1d`` can also be used from the command line to provide the most flexibility and ease of automation.

Key Components
--------------
Load the required modules

.. ipython:: python

    import matplotlib.pyplot as plt
    import pyrixs.process1d as p1d

Load spectra and plot them
--------------
The function ``process1d.load_spectra`` loads a list of spectra in text files, which can be found ``process1d.get_all_spectra_names``.

.. ipython:: python

    search_path='../notebooks/test_data/*.txt'
    spectra = p1d.load_spectra(search_path, p1d.get_all_spectra_names(search_path))
    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    artists = p1d.plot_spectra(ax1, spectra)

.. plot::

    import matplotlib.pyplot as plt
    import pyrixs.process1d as p1d
    search_path='../notebooks/test_data/*.txt'
    spectra = p1d.load_spectra(search_path, p1d.get_all_spectra_names(search_path))
    print(spectra)
    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    artists = p1d.plot_spectra(ax1, spectra)
    p1d.plt.show()



.. plot::
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.random.randn(1000)
    plt.hist( x, 20)
    plt.grid()
    plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
    plt.show()
