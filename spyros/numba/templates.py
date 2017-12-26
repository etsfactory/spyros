from numba import guvectorize
import numpy as np


NUMBA_SIGNATURE_LIST = ['void(int32[:],int32[:])',
                           'void(int64[:],int64[:])',
                           'void(float32[:],float32[:])',
                           'void(float64[:],float64[:])']
# Output type is void because gufuncs expect to be passed a pointer to the
# output array. That is also the reason why the type of the output is
# defined as a vector.
# Numba & numpy signature notation may not be the best. To be safe, just think of
# how you would do it in C.

REDUCE_NUMPY_SIGNATURE = '(n)->()'
RESIZE_NUMPY_SIGNATURE = '(n),(m)'
TRANSFORM_NUMPY_SIGNATURE = '(n)->(n)'

################################################################################
# Interesting links
################################################################################

# What is guvectorize?
# http://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html#numba.guvectorize
# What does nopython mean?
# http://numba.pydata.org/numba-doc/dev/glossary.html#term-nopython-mode

################################################################################
# Example of a reduce-type function
################################################################################

@guvectorize(NUMBA_SIGNATURE_LIST, REDUCE_NUMPY_SIGNATURE, nopython=False)
def _nansum(a, out):
    """Nansum docstring"""
    out[0] = 0
    for ai in a.flat:
        if np.isnan(ai):
            continue
        out[0] += ai


def nansum(arr, axis=0):
    arr = np.moveaxis(arr, axis, -1)  # transpose may be faster
    return _nansum(arr)


################################################################################
# Example of a resize-type function
################################################################################

@guvectorize(NUMBA_SIGNATURE_LIST, RESIZE_NUMPY_SIGNATURE, nopython=False)
def _foo(a, out):
    for i in range(len(a)):
        out[i] = a[i]


def foo(arr, axis=0):
    all_axes = [n for n in range(arr.ndim) if n != axis] + [axis]
    arr = arr.transpose(all_axes)
    new_shape = list(arr.shape)
    new_shape[-1] += 3  # alter dimensions however you want
    out = np.empty(new_shape)  # pre-allocate output array
    _foo(arr, out)  # fill output array according to computation
    out = np.moveaxis(out, -1, axis)
    return out


@guvectorize(NUMBA_SIGNATURE_LIST, RESIZE_NUMPY_SIGNATURE, nopython=False)
def _cum_returns(a, out):

    for i in range(len(a)):
        if not np.isnan(a[i]):
            break
        out[i] = np.nan

    crp1 = 1  # cumulative return plus 1
    nxt_out = 0  # crp1 - 1
    for j in range(i, len(a)):
        out[j] = nxt_out
        if not np.isnan(a[j]):
            crp1 *= 1 + a[j]
            nxt_out = crp1 - 1
        else:
            nxt_out = np.nan

    out[len(a)] = nxt_out


def cum_returns(arr):  # axis = 0 only
    arr = arr.T
    new_shape = list(arr.shape)
    new_shape[-1] += 1  # alter dimensions however you want
    out = np.empty(new_shape)  # pre-allocate output array
    _cum_returns(arr, out)  # fill output array according to computation
    return out.T


# Rolling (moving window)


# Expanding (growing window)


################################################################################
# Example of a transform-type function
################################################################################



################################################################################


if __name__ == "__main__":
    a = np.arange(12, dtype=np.float)
    A = np.reshape(a, (4, 3))
    out = nansum(A, axis=1)
    print(out)
    out = foo(A, axis=0)
    print(out)
    
    a = np.array([np.nan, 0.01, np.nan, np.nan, 0.02, -0.02, -0.01, np.nan, 0.01])
    out = np.empty(a.shape[0]+1)
    _cum_returns(a, out)
    print(np.round(out, 2))
