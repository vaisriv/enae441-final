# libs
from typing import NamedTuple
import numpy as np

class Measurement(NamedTuple):
    t: np.ndarray
    i: np.ndarray
    ρ: np.ndarray
    dρ: np.ndarray
