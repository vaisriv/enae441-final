# libs
import matplotlib.pyplot as plt
import numpy as np

# src
from measurements import *
from helpers import *

def a(oe: list, σ_n: list, σ: list, σ_a: float, μ: float) -> [np.ndarray, np.ndarray, float, np.ndarray]:
    # choose x0 from starting OE
    r0, v0 = system.coe2rv(oe, μ)
    X0_plus = np.hstack([r0, v0])

    # choose P0 from starting range variance
    σ_r, σ_v = [float(x)/σ_a for x in σ_n]
    P0_plus = np.diag([σ_r**2]*3 + [σ_v**2]*3)

    # choose R0 from measurement noise
    R0 = np.diag(σ_n)

    return [X0_plus, P0_plus, R0]

def b(data: measurements.Measurement, X0_plus: np.ndarray, P0_plus: np.ndarray, σ_a: float, μ: float) -> [list, np.ndarray, np.ndarray]:
    t_pred, X_minus_hist, P_minus_hist = propagators.kf(data, X0_plus, P0_plus, σ_a, μ)
    return t_pred, X_minus_hist, P_minus_hist

def c(t_pred: list, X_minus_hist: np.ndarray, P_minus_hist: np.ndarray) -> plt.Figure:
    σs = np.sqrt(np.clip(np.stack([np.diag(P) for P in P_minus_hist]), 0.0, np.inf))  # Nx6
    bounds = 3.0 * σs  # Nx6

    state_names = ["x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"]
    colors = {0: "r", 1: "g", 2: "b"}

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize=(12, 8))

    for j in range(len(state_names)):
        ax = axs[j%3, int(j/3)]
        ax.fill_between(t_pred,  bounds[:, j], -bounds[:, j],
                        color = colors.get(j%3),
                        alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"±3σ {state_names[j]}")

    fig.supxlabel("Time [s]")
    fig.supylabel("Error Bound (Centered at 0)")
    fig.suptitle("Pre-Update Prediction Covariance Bounds (±3σ)")
    fig.tight_layout()

    return fig
