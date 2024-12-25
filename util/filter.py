# util/filter.py
import numpy as np

def remove_zero(array):
    try:
        array[array == 0] = 1
    except TypeError:
        if array == 0:
            return 1
    return array

def threshold(array, threshold, set_val):
    temp = np.copy(array)
    temp[temp < threshold] = set_val
    return temp

def cap(array, threshold, set_val):
    temp = np.copy(array)
    temp[temp > threshold] = set_val
    return temp
