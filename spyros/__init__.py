import numpy as np
import bottleneck as bn

# Function are basically picked from distinct sources
# 1. Bottleneck functions
# Reference: http://berkeleyanalytics.com/bottleneck/reference.html
# 2. Numpy functions
# Reference: https://docs.scipy.org/doc/numpy/reference/routines.math.html
# 3. Those that are not available out there, require custom implementations:
# Take a look at https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#methods for ideas using numpy ufunc's

## Function implementation starts here
# Reduce functions
mean = bn.nanmean
std = bn.nanstd
var = bn.nanvar
sum = bn.nansum
max = bn.nanmax
min = bn.nanmin
argmax = bn.nanargmax
argmin = bn.nanargmin
median = bn.nanmedian
ss = bn.ss
anynan = bn.anynan
allnan = bn.allnan
	
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
	out = arr.copy()
	mask = np.isnan(arr)
	out[mask] = -np.Inf
	np.maximum.accumulate(out, axis=axis, out=out)
	out[out==-np.Inf]=np.nan
	return out

def cummin(arr, axis=0):
	out = arr.copy()
	mask = np.isnan(arr)
	out[mask] = np.Inf
	np.minimum.accumulate(out, axis=axis, out=out)
	out[out==np.Inf]=np.nan
	return out

