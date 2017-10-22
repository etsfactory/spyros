import numpy as np
import bottleneck as bn  # requires bottleneck >= 1.2.0
import pandas.core.nanops as nanops

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
movvar = bn.move_var
movmin = bn.move_min,
movmax = bn.move_max
movargmin = bn.move_argmin
movargmax = bn.move_argmax
movmedian = bn.move_median
movrank = bn.move_rank

# Expanding window functions
cumsum = np.nancumsum
cumprod = np.nancumprod


def cummax(arr, axis=0):
    """Cumulative max (in an expanding window).

    Parameters
    ----------
    arr : ND-array
    axis : int

    Returns
    -------
    cummax : ND-array
    """
    out = arr.copy()
    mask = np.isnan(arr)
    out[mask] = -np.Inf
    np.maximum.accumulate(out, axis=axis, out=out)
    out[out == -np.Inf] = np.nan
    return out


def cummin(arr, axis=0):
    """Cumulative min (in an expanding window).

    Parameters
    ----------
    arr : ND-array
    axis : int

    Returns
    -------
    cummin : ND-array
    """
    out = arr.copy()
    mask = np.isnan(arr)
    out[mask] = np.Inf
    np.minimum.accumulate(out, axis=axis, out=out)
    out[out == np.Inf] = np.nan
    return out
