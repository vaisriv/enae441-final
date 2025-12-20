# libs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# src
from measurements import *
from .system import *

def kf(data: measurements.Measurement, X0_plus: np.ndarray, P0_plus: np.ndarray, σ_a: float, μ: float) -> [list, np.ndarray, np.ndarray]:
    # extract arrays from data
    t = np.asarray(data.t, dtype=float)

    # shift epoch so t[0] = 0
    t0 = t.min()
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

def ekf(data: measurements.Measurement, X0_plus: np.ndarray, P0_plus: np.ndarray, R0: np.ndarray, stations: dict, σ_a: float, μ: float, RE: float, OMEGA_E: float, GAMMA0: float) -> [list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # extract arrays from data
    t = np.asarray(data.t, dtype=float)
    i = np.asarray(data.i, dtype=float)
    ρ = np.asarray(data.ρ, dtype=float)
    dρ = np.asarray(data.dρ, dtype=float)

    # station positions
    R_ecef = site_ecef(stations, RE)

    # shift epoch so t[0] = 0
    t0 = t.min()
    t = t - t0

    # time steps
    N = len(t)

    # sort by time (for propagation)
    order = np.argsort(t, kind="mergesort")
    t, i = t[order], i[order]
    ρ_meas, dρ_meas = ρ[order], dρ[order]

    # prediction storage
    X_minus_hist = np.zeros((N, 6))
    P_minus_hist = np.zeros((N, 6, 6))
    X_plus_hist = np.zeros((N, 6))
    P_plus_hist = np.zeros((N, 6, 6))

    yhat_minus_hist = np.zeros((N, 2))
    yhat_plus_hist  = np.zeros((N, 2))
    resid_pre_hist  = np.zeros((N, 2)) # y - h(X_minus)
    resid_post_hist = np.zeros((N, 2)) # y - h(X_plus)
    nis_pre_hist = np.zeros(N)

    X_plus = X0_plus.copy()
    P_plus = P0_plus.copy()
    t_prev = t[0]

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

            # process update
            Q_k = Q_discrete(dt, σ_a)
            P_minus = Φ @ P_plus @ Φ.T + Q_k
        else:
            # multiple measurements at same time tag: no propagation
            X_minus = X_plus
            P_minus = P_plus

        # save pre-update
        X_minus_hist[k, :] = X_minus
        P_minus_hist[k, :, :] = P_minus

        # measurement update
        y_k = np.array([ρ_meas[k], dρ_meas[k]])
        yhat_minus, H_k = meas_and_jacobian(X_minus, i[k], t_k, R_ecef, OMEGA_E, GAMMA0)
        yhat_minus_hist[k] = yhat_minus
        resid_pre_hist[k] = y_k - yhat_minus

        # innovation
        ν = y_k - yhat_minus
        S = H_k @ P_minus @ H_k.T + R0
        nis_pre = float(ν.T @ np.linalg.inv(S) @ ν)
        nis_pre_hist[k] = nis_pre

        # kalman gain
        # K = P_minus @ H_k.T @ np.linalg.pinv(S)
        K = P_minus @ H_k.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        # update state
        X_plus = X_minus + K @ ν

        # joseph covariance update
        I6 = np.eye(6)
        P_plus = (I6 - K @ H_k) @ P_minus @ (I6 - K @ H_k).T + K @ R0 @ K.T

        # save post-update
        X_plus_hist[k, :] = X_plus
        P_plus_hist[k, :, :] = P_plus

        # post-fit measurement
        yhat_plus, _ = meas_and_jacobian(X_plus, i[k], t_k, R_ecef, OMEGA_E, GAMMA0)
        yhat_plus_hist[k] = yhat_plus
        resid_post_hist[k] = y_k - yhat_plus

        # advance time
        t_prev = t_k

    return t, X_minus_hist, P_minus_hist, X_plus_hist, P_plus_hist, yhat_minus_hist, yhat_plus_hist, resid_pre_hist, resid_post_hist, nis_pre_hist
