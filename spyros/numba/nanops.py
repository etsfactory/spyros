import numpy as np
from numba import vectorize, guvectorize, int32, int64, float32, float64

# http://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html#vectorized-functions-ufuncs-and-dufuncs
@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
def nansum(a, b):
    return a + b

nancumsum = nansum.accumulate


##################################### Other interesting approach #########################################
from numbagg.decorators import ndreduce, ndmoving # pip install git+https://github.com/shoyer/numbagg.git

@ndmoving
def _nancumlast(a, window, out=None):
    """nancumlast"""
    val = np.nan
    for i in range(len(a)):
        ai = a[i]
        if np.isnan(ai):
            ai = val
        else:
            val = ai
        out[i] = ai
        
def nancumlast(a, axis=None):
    out = _nancumlast(a, window=1, axis=axis)
    return out.T if axis==0 else out
    
@ndmoving
def _nancumsum(a, window, out=None):
    """nancumsum"""
    asum = 0
    for i in range(len(a)):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
        out[i] = asum
        
def nancumsum(a, axis=0):
    out = _nancumsum(a, window=1, axis=axis)
    return out.T if axis==0 else out
