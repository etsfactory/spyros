import numpy as np
import bottleneck as bn  # requires bottleneck >= 1.2.0
import pandas.core.nanops as nanops

# Functions in this module are basically picked from distinct sources:
# 1. Bottleneck functions
# Reference: http://berkeleyanalytics.com/bottleneck/reference.html
# 2. Numpy functions
# Reference: https://docs.scipy.org/doc/numpy/reference/routines.math.html
# 3. Pandas nanops module functions (for many, it is just a wrapper of bottleneck)
# Reference: https://github.com/pandas-dev/pandas/blob/master/pandas/core/nanops.py
# 4. Scipy.stats operations for unmasked and masked arrays
# - https://docs.scipy.org/doc/scipy-0.16.0/reference/stats.html#statistical-functions
# - https://docs.scipy.org/doc/scipy-0.16.1/reference/stats.mstats.html
# END. Those that are not available out there, require custom implementations:
# Take a look at https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#methods for ideas using numpy ufunc's

## Function implementation starts here
# Reducing functions
mean = bn.nanmean
std = bn.nanstd
var = bn.nanvar
sum = bn.nansum
prod = nanops.nanprod
max = bn.nanmax
min = bn.nanmin
argmax = bn.nanargmax
argmin = bn.nanargmin
median = bn.nanmedian
ss = bn.ss
anynan = bn.anynan
allnan = bn.allnan
sem = nanops.nansem
skew = nanops.nanskew
kurt = nanops.nankurt
percentile = np.nanpercentile

# Reducing two-argument functions
cov = nanops.nancov
corr = nanops.nancorr

# Element-wise functions
gt = nanops.nangt
ge = nanops.nange
lt = nanops.nanlt
le = nanops.nanle
eq = nanops.naneq
ne = nanops.nanne

# Moving window functions
movsum = bn.move_sum
movmean = bn.move_mean
movstd = bn.move_std
movmin = bn.move_min
movmax = bn.move_max
movmedian = bn.move_median

# Expanding window functions
cumsum = np.nancumsum
cumprod = np.nancumprod


def cummax(arr, axis=0):
    out = arr.copy()
    mask = np.isnan(arr)
    out[mask] = -np.Inf
    np.maximum.accumulate(out, axis=axis, out=out)
    out[out == -np.Inf] = np.nan
    return out


def cummin(arr, axis=0):
    out = arr.copy()
    mask = np.isnan(arr)
    out[mask] = np.Inf
    np.minimum.accumulate(out, axis=axis, out=out)
    out[out == np.Inf] = np.nan
    return out
