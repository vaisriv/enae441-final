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
import p05

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
        0.6, # [-]
        np.deg2rad(45), # [deg -> rad]
        np.deg2rad(0), # [deg -> rad]
        np.deg2rad(170), # [deg -> rad]
        np.deg2rad(45) # [deg -> rad]
    ]
    stations = {
        0: (np.deg2rad(35.297), np.deg2rad(-116.914)), # [lat, long]
        1: (np.deg2rad(40.4311), np.deg2rad(-4.248)), # [lat, long]
        2: (np.deg2rad(-35.4023), np.deg2rad(148.9813)), # [lat, long]
    }
    # measurement noise
    σ_n = [
        1e-6, # [km^2]
        1e-10 # [km^2/s^2]
    ]
    # process noise tuning
    σ = [
        1e-2, # [km]
        1e-4 # [km/s]
    ]
    σ_a = 1e-4 # [km/s^2]

    # load numpy measurements
    data = numpy_data.load("./data/Project-Measurements-Hard.npy")

    #######
    # p01 #
    #######
    # p01e
    p01.e(data).savefig("./outputs/figures/s07-01e.png")

    #######
    # p03 #
    #######
    # p03a
    X0_plus, P0_plus, R0 = p03.a(oe, σ_n, σ, σ_a, MU)
    with open("./outputs/text/s07-03a.txt", "w", encoding="utf-8") as f:
        f.write(f"X₀⁺ =\n{X0_plus}\n")
        f.write(f"P₀⁺ =\n{P0_plus}\n")
        f.write(f"R₀ =\n{R0}")
    # p03b
    t_pred, X_minus_hist, P_minus_hist = p03.b(data, X0_plus, P0_plus, σ_a, MU)
    # p03c
    p03.c(t_pred, X_minus_hist, P_minus_hist).savefig("./outputs/figures/s07-03c.png")

    #######
    # p04 #
    #######
    # p04a
    t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist, yhat_minus_hist, yhat_plus_hist, resid_pre_hist, resid_post_hist, nis_pre_hist = p04.a(data, X0_plus, P0_plus, R0, stations, σ_a, MU, RE, OMEGA_E, GAMMA0)
    # p04b
    p04.b(t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist).savefig("./outputs/figures/s07-04b.png")
    # p04c
    p04.c(t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist).savefig("./outputs/figures/s07-04c.png")

    #######
    # p05 #
    #######
    # p05a
    resid_fig = p05.a(data, t_pred, resid_pre_hist, resid_post_hist)
    resid_axs = resid_fig.get_axes()
    for ax in resid_axs:
        ax.set_yscale("log")
    resid_fig.savefig("./outputs/figures/s07-05a.png")
    # p05b
    rms_r_m, rms_rr_cms, nis_mean = p05.b(resid_post_hist, nis_pre_hist)
    with open("./outputs/text/s07-05b.txt", "w", encoding="utf-8") as f:
        f.write(f"Post-fit RMS range residual: {rms_r_m:.3f} [m]\n")
        f.write(f"Post-fit RMS range-rate residual: {rms_rr_cms:.3f} [cm/s]\n")
        f.write(f"Mean pre-fit NIS (df=2): {nis_mean:.3f}\n")
    # p05c
    p05.c(t_pred, X_plus_hist, P_plus_hist).savefig("./outputs/figures/s07-05c.png")
    # p05d
    X_final, σ_final, names = p05.d(X_plus_hist, P_plus_hist)
    with open("./outputs/text/s07-05d.txt", "w", encoding="utf-8") as f:
        f.write("Final state estimate (ECI, km and km/s):")
        for name, X, σ in zip(names, X_final, σ_final):
            f.write(f"{name:8s}: {X: .9f} ± {σ: .9f} (1σ)\n")

    return 0


if __name__ == "__main__":
    main()
