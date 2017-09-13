import bottleneck as bn

# First import some bottleneck functions
# See http://berkeleyanalytics.com/bottleneck/reference.html# for more
nanmean = bn.nanmean
nanstd = bn.nanstd
nansum = bn.nansum
nanmax = bn.nanmax
nanmin = bn.nanmin
nanargmax = bn.nanargmax
nanargmin = bn.nanargmin

# Others, not available out there, require custom implementations
# Take a look at https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#methods
# nancummax (np.maximum.accumulate + masking?)
# nancummin (np.minimum.accumulate + masking?)

