# libs
import numpy as np
import os

# src
from .measurements import *

def raw_load(file_path: str) -> np.ndarray:
    # cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    # full_path = cur_dir + file_path
    full_path = file_path

    try:
        raw_data = np.load(full_path, allow_pickle=True)
    except NameError:
        print(f"Could not load data from {file_path}")

    return raw_data

def parse(raw_data: np.ndarray) -> Measurement:
    data = Measurement(raw_data[:, 0], raw_data[:, 1], raw_data[:, 2], raw_data[:, 3])

    return data

def load(file_path: str) -> Measurement:
    raw_data = raw_load(file_path)
    data = parse(raw_data)

    return data
