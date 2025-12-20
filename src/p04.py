# libs
import matplotlib.pyplot as plt
import numpy as np

# src
from measurements import *
from helpers import *

def a(data: measurements.Measurement, X0_plus: np.ndarray, P0_plus: np.ndarray, R0: np.ndarray, stations: dict, σ_a: float, μ: float, RE: float, OMEGA_E: float, GAMMA0: float) -> [list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist = propagators.ekf(data, X0_plus, P0_plus, R0, stations, σ_a, μ, RE, OMEGA_E, GAMMA0)
    return t_pred, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist

def b(t_pred: list, X_minus_hist: np.ndarray, P_minus_hist: np.ndarray, X_plus_hist: np.ndarray, P_plus_hist: np.ndarray) -> plt.Figure:
    σ_ms = np.sqrt(np.clip(np.stack([np.diag(P) for P in P_minus_hist]), 0.0, np.inf))  # Nx6
    σ_ps = np.sqrt(np.clip(np.stack([np.diag(P) for P in P_plus_hist]), 0.0, np.inf))  # Nx6
    bound_ms = 3.0 * σ_ms # Nx6
    bound_ps = 3.0 * σ_ps # Nx6

    state_names = ["x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"]
    colors = {0: "r", 1: "g", 2: "b"}

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize=(12, 8))

    for j in range(len(state_names)):
        ax = axs[j%3, int(j/3)]
        ax.fill_between(t_pred,  bound_ms[:, j], -bound_ms[:, j],
                        color = "gray",
                        alpha=0.5)
        ax.fill_between(t_pred,  bound_ps[:, j], -bound_ps[:, j],
                        color = colors.get(j%3),
                        alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"±3σ {state_names[j]}")

    plt.figlegend(["Pre-Update", "Post-Update"])
    fig.supxlabel("Time [s]")
    fig.supylabel("Error Bound (Centered at 0)")
    fig.suptitle("Prediction Covariance Bounds (±3σ)")
    fig.tight_layout()

    return fig

def c(t_pred: list, X_minus_hist: np.ndarray, P_minus_hist: np.ndarray, X_plus_hist: np.ndarray, P_plus_hist: np.ndarray) -> plt.Figure:
    dX = (X_plus_hist - X_minus_hist)/1e6 # Nx6
    σ_ms = np.sqrt(np.clip(np.stack([np.diag(P) for P in P_minus_hist]), 0.0, np.inf))  # Nx6
    σ_ps = np.sqrt(np.clip(np.stack([np.diag(P) for P in P_plus_hist]), 0.0, np.inf))  # Nx6
    bound_ms = 3.0 * σ_ms # Nx6
    bound_ps = 3.0 * σ_ps # Nx6

    state_names = ["x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"]
    colors = {0: "r", 1: "g", 2: "b"}

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize=(12, 8))

    for j in range(len(state_names)):
        ax = axs[j%3, int(j/3)]
        ax.fill_between(t_pred,  bound_ms[:, j], -bound_ms[:, j],
                        color = "gray",
                        alpha=0.5)
        ax.fill_between(t_pred,  bound_ps[:, j], -bound_ps[:, j],
                        color = "dimgray",
                        alpha=0.5)
        ax.plot(t_pred, dX,
                linestyle="-", linewidth=0,
                marker=".", markersize=2,
                color = colors.get(j%3))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"±3σ {state_names[j]}")

    plt.figlegend(["Pre-Update Bounds", "Post-Update Bounds", "Differences"])
    fig.supxlabel("Time [s]")
    fig.supylabel("Prediction (Centered at 0)")
    fig.suptitle("Δμₓ,ₖ Between Prediction Covariance Bounds (±3σ)")
    fig.tight_layout()

    return fig
