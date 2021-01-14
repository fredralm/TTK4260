#!/usr/bin/env Python
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
from queue import Queue

def remove_dict_keys(data: Dict, to_remove: List):
    ''' Remove the dictionary entries based on the entry key. '''
    for key in to_remove:
        del data[key]
    return data

def remove_array_entries(arr, indices):
    ''' Remove the array entries with the given indices.
    arg arr: Nx1 array of floats
    arg indices: list of ints '''
    n = arr.shape[0]
    mask = np.full(arr.shape, True, dtype=bool)
    for i in indices:
        if i >= n or i < 0:
            continue
        else:
            mask[i,:] = False

    return arr[mask]

def print_dict_entries(x: Dict):
    ''' Prints the key and value type for every dictionary entry. '''
    for key, value in x.items():
        print('key: {}, value: {}'.format(key, type(value)))

def print_array_info(x: np.ndarray):
    ''' Prints the shape and data type of a numpy array. '''
    print('shape: {}, dtype: {}'.format(x.shape, x.dtype))


