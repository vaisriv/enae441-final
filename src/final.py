# libs
import matplotlib.pyplot as plt
import numpy as np
import os

# src
from measurements import *
import p01

def main() -> int:
    data = numpy_data.load("./data/Project-Measurements-Easy.npy")

    #######
    # p01e #
    #######
    p01.e(data, "./outputs/figures/s01e.png")

    return 0


if __name__ == "__main__":
    main()
