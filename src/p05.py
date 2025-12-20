# libs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# src
from measurements import *
from helpers import *

def a(data: measurements.Measurement, t: list, resid_pre_hist: np.ndarray, resid_post_hist: np.ndarray) -> plt.Figure:
    # extract arrays from data
    t = np.asarray(data.t, dtype=float)
    i = np.asarray(data.i, dtype=float)

    # shift epoch so t[0] = 0
    t0 = t.min()
    t = t - t0

    # sort by time
    order = np.argsort(t, kind="mergesort")
    t, i = t[order], i[order]

    # shift units for display
    res_pre_r_m = resid_pre_hist[:, 0] * 1e3 # [km] -> [m]
    res_post_r_m = resid_post_hist[:, 0] * 1e3 # [km] -> [m]
    res_pre_rr_cms = resid_pre_hist[:, 1] * 1e5 # [km/s] -> [cm/s]
    res_post_rr_cms = resid_post_hist[:, 1] * 1e5 # [km/s] -> [cm/s]

    # station info
    labels = {0: "DSN #0 Goldstone", 1: "DSN #1 Madrid", 2: "DSN #2 Canberra"}
    colors = {0: "r", 1: "g", 2: "b"}

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(12, 8))

    for stn in np.unique(i):
        m = (i == stn)

        # range residuals
        # axs[0].plot(t[m], res_pre_r_m[m],
        #         linestyle="-", linewidth=0,
        #         marker=".", markersize=2,
        #         color=colors.get(int(stn)),
        #         label=labels.get(int(stn), f"DSN #{int(stn)}"))
        axs[0].plot(t[m], res_post_r_m[m],
                linestyle="-", linewidth=0,
                marker=".", markersize=2,
                color=colors.get(int(stn)),
                label=labels.get(int(stn), f"DSN #{int(stn)}"))
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc = "upper right")
        axs[0].set_ylabel("Range Residuals [km]")

        # range-rate residuals
        # axs[1].plot(t[m], res_pre_rr_cms[m],
        #         linestyle="-", linewidth=0,
        #         marker=".", markersize=2,
        #         color=colors.get(int(stn)),
        #         label=labels.get(int(stn), f"DSN #{int(stn)}"))
        axs[1].plot(t[m], res_post_rr_cms[m],
                linestyle="-", linewidth=0,
                marker=".", markersize=2,
                color=colors.get(int(stn)),
                label=labels.get(int(stn), f"DSN #{int(stn)}"))
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc = "upper right")
        axs[1].set_ylabel("Range-Rate Residuals [km/s]")

    # measurement noise reference bands
    # axs[0].fill_between(t, -3.0, +3.0,
    #                     color = "gray",
    #                     alpha=0.5)
    # axs[1].fill_between(t, -3.0, +3.0,
    #                     color = "gray",
    #                     alpha=0.5)

    fig.supxlabel("Time [s]")
    fig.suptitle("DSN Measurement Residuals vs. Time [s]")
    fig.tight_layout()

    return fig

def b(resid_post_hist: np.ndarray, nis_pre: np.ndarray) -> [float, float, float]:
    # shift units
    res_post_r_m = resid_post_hist[:, 0] * 1e3 # [km] -> [m]
    res_post_rr_cms = resid_post_hist[:, 1] * 1e5 # [km/s] -> [cm/s]

    rms_r_m = np.sqrt(np.mean(res_post_r_m**2))
    rms_rr_cms = np.sqrt(np.mean(res_post_rr_cms**2))
    nis_mean = np.mean(nis_pre)

    return rms_r_m, rms_rr_cms, nis_mean

def c(t_pred: list, X_plus_hist: np.ndarray, P_plus_hist: np.ndarray) -> plt.Figure:
    σ_ps = np.sqrt(np.clip(np.stack([np.diag(P) for P in P_plus_hist]), 0.0, np.inf))  # Nx6
    bound_ps = 3.0 * σ_ps # Nx6

    state_names = ["x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"]
    colors = {0: "r", 1: "g", 2: "b"}

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize=(12, 8))

    for j in range(len(state_names)):
        ax = axs[j%3, int(j/3)]
        ax.fill_between(t_pred, X_plus_hist[:, j] - bound_ps[:, j], X_plus_hist[:, j] + bound_ps[:, j],
                        color = "dimgray",
                        alpha=0.5)
        ax.plot(t_pred, X_plus_hist[:, j],
                linestyle="-", linewidth=0,
                marker=".", markersize=2,
                color = colors.get(j%3))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{state_names[j]}")

    plt.figlegend(["±3σ Bounds", "Estimate"])
    fig.supxlabel("Time [s]")
    fig.supylabel("State Estimates")
    fig.suptitle("Final State Estimates (ECI, [km] & [km/s])")
    fig.tight_layout()

    return fig

def d(X_plus_hist: np.ndarray, P_plus_hist: np.ndarray) -> [np.ndarray, np.ndarray]:
    X_final = X_plus_hist[-1]
    σ_final = np.sqrt(np.clip(np.diag(P_plus_hist[-1]), 0.0, np.inf)) # 1σ
    state_names = ["x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"]

    return X_final, σ_final, state_names
