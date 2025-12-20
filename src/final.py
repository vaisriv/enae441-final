# libs
import matplotlib.pyplot as plt
import numpy as np
import os

# src
from measurements import *
from helpers import *
import p01
import p03
import p04

def main() -> int:
    # constants
    EPS = 1e-12 # [-]
    MU = 398600.4418 # [km^3/s^2]
    RE = 6378.137 # [km]
    OMEGA_E = 7.292115e-5 # [rad/s]
    GAMMA0 = 0.0 # [rad]

    # givens
    oe = [
        7e3, # [km]
        0.2, # [-]
        np.deg2rad(45), # [deg -> rad]
        np.deg2rad(0), # [deg -> rad]
        np.deg2rad(270), # [deg -> rad]
        np.deg2rad(78.75) # [deg -> rad]
    ]
    σ = [
        1e-6, # [km^2]
        1e-10 # [km^2/s^2]
    ]
    stations = {
        0: (np.deg2rad(35.297), np.deg2rad(-116.914)), # [lat, long]
        1: (np.deg2rad(40.4311), np.deg2rad(-4.248)), # [lat, long]
        2: (np.deg2rad(-35.4023), np.deg2rad(148.9813)), # [lat, long]
    }

    # process noise tuning
    σ_a = 1e-6 # [km/s^2]

    # load numpy measurements
    data = numpy_data.load("./data/Project-Measurements-Easy.npy")

    #######
    # p01 #
    #######
    # p01e
    p01.e(data).savefig("./outputs/figures/s01e.png")

    #######
    # p03 #
    #######
    # p03a
    X0_plus, P0_plus, R0 = p03.a(oe, σ, σ_a, MU)
    with open("./outputs/text/s03a.txt", "w", encoding="utf-8") as f:
        f.write(f"X₀⁺ =\n{X0_plus}\n")
        f.write(f"P₀⁺ =\n{P0_plus}\n")
        f.write(f"R₀ =\n{R0}")
    # p03b
    t_pred, X_minus_hist, P_minus_hist = p03.b(data, X0_plus, P0_plus, σ_a, MU)
    # p03c
    p03.c(t_pred, X_minus_hist, P_minus_hist).savefig("./outputs/figures/s03c.png")

    #######
    # p04 #
    #######
    # p04a
    t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist = p04.a(data, X0_plus, P0_plus, R0, stations, σ_a, MU, RE, OMEGA_E, GAMMA0)
    # p04b
    p04.b(t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist).savefig("./outputs/figures/s04b.png")
    # p04c
    p04.c(t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist).savefig("./outputs/figures/s04c.png")

    return 0


if __name__ == "__main__":
    main()
