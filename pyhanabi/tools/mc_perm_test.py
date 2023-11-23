import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy
from datetime import datetime
import numpy as np
import random
from scipy import stats


def mc_perm_test():
    br_sad_six = [
        12.75642857142857,
        13.030285714285716,
        11.858000000000002,
        12.142428571428573,
        11.457571428571427,
        14.485000000000001,
        11.769142857142857,
        14.233857142857143,
        16.57957142857143,
        12.96685714285714
    ]

    ch_e50_sad_six = [
        12.7564,
        11.944857142857144,
        12.017285714285714,
        11.891857142857143,
        11.335999999999999,
        14.07414285714286,
        11.531714285714285,
        13.640857142857143,
        15.555142857142856,
        13.415142857142857
    ]


    sba_sad_six = np.mean([
        [20.78, 19.862, 11.938, 3.436, 19.738, 4.008, 18.817],
        [16.655, 20.177, 13.701, 20.981, 20.83, 4.907, 0.116],
        [21.879, 21.374, 14.508, 21.632, 3.708, 20.603, 0.071],
        [21.149, 16.866, 13.006, 21.584, 4.286, 4.809, 20.538],
        [22.114, 16.099, 14.829, 21.73, 4.171, 21.392, 0.032],
        [21.85, 16.437, 18.03, 9.633, 19.558, 19.075, 21.606],
        [21.443, 17.041, 16.074, 18.363, 2.538, 20.889, 3.279],
        [16.361, 18.072, 19.711, 19.034, 2.476, 20.938, 20.734],
        [20.661, 16.648, 20.4, 21.483, 20.62, 4.44, 19.655],
        [21.452, 21.453, 15.532, 20.133, 21.706, 2.961, 5.341]
    ], axis=1)

    ch_e50_sba_sad_six = np.mean([
        [20.288, 20.432, 15.201, 4.054, 20.096, 4.383, 18.773],
        [15.943, 20.742, 14.148, 20.626, 20.256, 4.09, 0.03],
        [21.635, 21.875, 14.802, 21.843, 3.563, 21.357, 0.058],
        [20.521, 16.662, 13.871, 21.435, 3.546, 4.336, 19.947],
        [22.119, 15.432, 14.306, 21.213, 3.193, 21.214, 0.031],
        [21.855, 15.668, 17.728, 13.643, 19.142, 18.97, 21.591],
        [22.113, 17.747, 16.736, 19.115, 2.7, 21.703, 4.475],
        [16.52, 18.567, 18.863, 19.036, 3.627, 21.16, 21.108],
        [21.022, 16.209, 20.003, 20.36, 20.738, 3.387, 19.425],
        [21.898, 21.015, 17.505, 20.36, 21.505, 3.892, 5.269]
    ], axis=1)

    br_sad_six_vs_iql = [
        15.220666666666668,
        16.349666666666668,
        15.216416666666666,
        15.180416666666668,
        15.65791666666667,
        13.360166666666666,
        14.788916666666667,
        14.724750000000002,
        16.05,
        15.590083333333334
    ]

    sba_sad_six_vs_iql = [
        16.352416666666667,
        16.950916666666668,
        17.562833333333334,
        17.29216666666667,
        17.690749999999998,
        16.032833333333333,
        15.547333333333333,
        15.888166666666669,
        16.966333333333335,
        16.839666666666666
    ]

    
    print("ch_e50_sba_sad_six > sba_sad_six")
    generate_mcpt_table(ch_e50_sba_sad_six, sba_sad_six)

    print("ch_e50_sba_sad_six > sba_sad_six")
    generate_mcpt_table(ch_e50_sba_sad_six, sba_sad_six)

    print("sba_sad_six_vs_iql > br_sad_six_vs_iql")
    generate_mcpt_table(ch_e50_sba_sad_six, sba_sad_six)


def generate_mcpt_table(greater, smaller, n_resamples=100000):
    def statistic(x, y, axis=0):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = stats.permutation_test(
        [greater, smaller],
        statistic,
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative="greater",
    )

    print(result.pvalue)


if __name__ == "__main__":
    mc_perm_test()
