import sys
import numpy as np
import pandas as pd
import math
import numba
from tqdm import tqdm, trange
from warnings import warn


def gaussian(t0, t, density, binwidth, epsilon=0.1, params=None):
    """ Returns weights for samples at times t for a Gaussian kernel centered at t0 """
    distance  = t-t0
    K = -np.log(epsilon) / ((binwidth) ** 2)
    return np.exp(-K * (distance * density) ** 2)

def square(t0, t, density, binwidth, epsilon=0.1, params=None):
    """ Returns weights for samples at times t for a square kernel centered at t0 """
    distance = t - t0
    out = (np.abs(distance) < (binwidth / density)).astype(int)
    return out

def triangle(t0, t, density, binwidth, epsilon=0.1, params=None):
    """ Returns weights for samples at times t for a triangle kernel centered at t0 """
    distance = np.abs(t - t0)
    effective_width = binwidth / (2 * density)
    out = np.where(distance < effective_width, 1 - distance / effective_width, 0)
    return out