import numpy as np

def nan_randn(num_rows, num_cols, nan_prob):
    x = np.random.randn(num_rows, num_cols)
    mask = np.random.rand(num_rows, num_cols) > (1 - nan_prob)
    x[mask] = np.nan
    return x
