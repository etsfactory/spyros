from numba import guvectorize
import numpy as np

REDUCE_NUMBA_SIGNATURE_LIST = ['void(int32[:],int32[:])',
                           'void(int64[:],int64[:])',
                           'void(float32[:],float32[:])',
                           'void(float64[:],float64[:])']
REDUCE_NUMPY_SIGNATURE = '(n)->()'

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

@guvectorize(REDUCE_NUMBA_SIGNATURE_LIST, REDUCE_NUMPY_SIGNATURE, nopython=False)
def _nansum(a, out):
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
