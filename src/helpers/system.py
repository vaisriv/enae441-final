# libs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# src
from measurements import *

def coe2rv(oe: list, μ:float) -> [np.ndarray, np.ndarray]:
    # a in km, angles in rad
    a, e, i, ω, Ω, ν = [float(x) for x in oe]

    # Semi-latus rectum
    p = a * (1 - e**2)

    # Perifocal coordinates (PQW frame)
    r_pf = (p / (1 + e*np.cos(ν))) * np.array([np.cos(ν), np.sin(ν), 0.0])
    v_pf = np.sqrt(μ/p) * np.array([-np.sin(ν), e + np.cos(ν), 0.0])

    # Rotation from PQW -> ECI (3-1-3 sequence)
    R_pqw_eci = sp.spatial.transform.Rotation.from_euler("ZXZ", [-ω, -i, -Ω])
    r = r_pf @ R_pqw_eci.as_matrix()
    v = v_pf @ R_pqw_eci.as_matrix()

    return r, v

def A_matrix(r: np.ndarray, μ: float) -> np.ndarray:
    rnorm = np.linalg.norm(r)
    I3 = np.eye(3)
    dadr = -μ * (I3/(rnorm**3) - 3.0*np.outer(r, r)/(rnorm**5))
    A = np.block([
        [np.zeros((3,3)), I3],
        [dadr, np.zeros((3,3))]
    ])

    return A

def Q_discrete(dt: float, σ_a: float) -> np.ndarray:
    if dt <= 0.0:
        return np.zeros((6,6))

    q_a = σ_a**2
    I3 = np.eye(3)

    Q = q_a * np.block([
        [(dt**3/3.0)*I3, (dt**2/2.0)*I3],
        [(dt**2/2.0)*I3, (dt)*I3]
    ])

    return Q

def eom_with_stm(t: float, y: list, μ: float) -> np.ndarray:
    # y = [r(3), v(3), Phi:flat(36)]
    r = y[0:3]
    v = y[3:6]
    Φ = y[6:].reshape(6,6)

    rnorm = np.linalg.norm(r)
    a = -μ * r / (rnorm**3)

    A = A_matrix(r, μ)
    dΦ = A @ Φ

    return np.concatenate([v, a, dΦ.reshape(-1)])

def site_ecef(stations: dict, RE: float) -> dict:
    R_ecef = {}

    for idx, (ϕ, λ) in stations.items():
        R_ecef[idx] = RE * np.array([
            np.cos(ϕ)*np.cos(λ),
            np.cos(ϕ)*np.sin(λ),
            np.sin(ϕ)
        ])

    return R_ecef

def site_eci(station_idx: int, t: float, R_ecef: dict, OMEGA_E: float, GAMMA0: float) -> [np.ndarray, np.ndarray]:
    γ = GAMMA0 + OMEGA_E * t
    ω = np.array([0.0, 0.0, OMEGA_E])

    R_ecef_eci = sp.spatial.transform.Rotation.from_euler("Z", -γ)
    R_site = R_ecef_eci.as_matrix() @ R_ecef[int(station_idx)]

    dR_site = np.cross(ω, R_site)

    return R_site, dR_site

def meas_and_jacobian(X: list, station_idx: int, t: float, R_ecef: dict, OMEGA_E: float, GAMMA0: float) -> [np.ndarray, np.ndarray]:
    r = X[0:3]
    v = X[3:6]
    R_site, dR_site = site_eci(station_idx, t, R_ecef, OMEGA_E, GAMMA0)

    ρ_vec = r - R_site
    ρ = np.linalg.norm(ρ_vec)
    u = ρ_vec / ρ

    v_rel = v - dR_site
    dρ = u @ v_rel

    yhat = np.array([ρ, dρ])
    I3 = np.eye(3)

    # H (2x6)
    H_ρ_r = u.reshape(1,3)
    H_ρ_v = np.zeros((1,3))
    H_dρ_v = u.reshape(1,3)
    H_dρ_r = (1.0/ρ) * (v_rel.reshape(1,3) @ (I3 - np.outer(u, u)))

    H = np.block([
        [H_ρ_r,  H_ρ_v],
        [H_dρ_r, H_dρ_v],
    ])

    return yhat, H
