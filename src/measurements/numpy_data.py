# libs
import numpy as np

# src
from .measurements import *

def raw_load(file_path: str) -> np.ndarray:
    try:
        raw_data = np.load(file_path, allow_pickle=True)
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
