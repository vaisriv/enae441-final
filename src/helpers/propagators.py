# libs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# src
from measurements import *
from .system import *

def predict_only(data: measurements.Measurement, X0_plus: np.ndarray, P0_plus: np.ndarray, σ_a: float, μ: float) -> [list, np.ndarray, np.ndarray]:
    # extract arrays from data
    t = np.asarray(data.t, dtype=float)

    # shift epoch so t[0] = 0
    t0 = t[0]
    t = t - t0

    # sort by time (for propagation)
    order = np.argsort(t)
    t = t[order]

    # prediction storage
    X_minus_hist = np.zeros((len(t), 6))
    P_minus_hist = np.zeros((len(t), 6, 6))

    X_plus = X0_plus.copy()
    P_plus = P0_plus.copy()
    t_prev = 0.0

    for k, t_k in enumerate(t):
        dt = t_k - t_prev

        if dt > 0:
            Φ0 = np.eye(6)
            y0 = np.concatenate([X_plus, Φ0.reshape(-1)])

            sol = sp.integrate.solve_ivp(
                fun=lambda tt, yy: eom_with_stm(tt, yy, μ),
                t_span=(t_prev, t_k),
                y0=y0,
                t_eval=[t_k],
                rtol=1e-10,
                atol=1e-12,
                method="DOP853",
            )
            y_k = sol.y[:, -1]
            X_minus = y_k[0:6]
            Φ = y_k[6:].reshape(6,6)

            # prediction-only, so no process update:
            Q_k = Q_discrete(dt, σ_a)
            P_minus = Φ @ P_plus @ Φ.T + Q_k
        else:
            # multiple measurements at same time tag: no propagation
            X_minus = X_plus
            P_minus = P_plus

        X_minus_hist[k, :] = X_minus
        P_minus_hist[k, :, :] = P_minus

        # prediction-only, so no measurement update:
        X_plus = X_minus
        P_plus = P_minus
        t_prev = t_k

    return t, X_minus_hist, P_minus_hist
