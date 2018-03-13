import numpy as np
from numba import guvectorize

NUMBA_SIGNATURE_LIST = ['void(int32[:],float32[:])',
                           'void(int64[:],float64[:])',
                           'void(float32[:],float32[:])',
                           'void(float64[:],float64[:])']
REDUCE_NUMPY_SIGNATURE = '(n)->()'


@guvectorize(NUMBA_SIGNATURE_LIST, REDUCE_NUMPY_SIGNATURE, nopython=True)
def _argfirst(a, out):
    out[0] = np.nan
    for i in range(len(a)):
        if not np.isnan(a[i]):
            out[0] = i
            break


def argfirst(arr, axis=0):
    arr = np.moveaxis(arr, axis, -1)  # transpose may be faster
    return _argfirst(arr)


@guvectorize(NUMBA_SIGNATURE_LIST, REDUCE_NUMPY_SIGNATURE, nopython=True)
def _arglast(a, out):
    out[0] = np.nan
    for i in range(len(a) - 1, -1, -1):
        if not np.isnan(a[i]):
            out[0] = i
            break


def arglast(arr, axis=0):
    arr = np.moveaxis(arr, axis, -1)  # transpose may be faster
    return _arglast(arr)
