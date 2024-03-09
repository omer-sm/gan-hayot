import numpy as np
import sys

VAE_LAYER_N = 512

def make_single_vector(x1, x2, x3):
    v1 = np.linspace(x1, x2, VAE_LAYER_N//2)
    v2 = np.linspace(x2, x3, VAE_LAYER_N//2)
    return np.concatenate([v1, v2])

def make_mat_from_vectors(v1, v2, m):
    return np.linspace(v1, v2, m).T

def make_multiple_vectors(x1, x2, x3, x4, x5, x6, m):
    v1 = make_single_vector(x1, x2, x3)
    v2 = make_single_vector(x4, x5, x6)
    return np.linspace(v1, v2, m).T

print(sys.argv[2:4])
#print(make_multiple_vectors(0.1, 1, 0.4, 0.4, 0.2, 0.1, 10))