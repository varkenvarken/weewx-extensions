Some weewx extensions, very much a work in progress

sunmoon.py:

    Makes sun azimuth and direction available for graphing if the sun is above the horizon.
    
    It is an exercise to make an new Xtype and service. Documentation in the source.

polarplots.py:

    A PolarPlotGenerator. Allows you to create polar plots of quantities that have a
    direction component, like winddir and windvec. Currently implemented plot_type options
    
    - vector_dir
        time v direction part of vector and color of dots determined by magnitude of the vector
    - histogram
        histogram of direction
    - histogram_vector_dir
        histogram of the size of vector quantity
        this produces a wind rose

