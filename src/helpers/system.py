# libs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# src
from measurements import *

def coe2rv(oe: list, μ:float) -> [np.ndarray, np.ndarray]:
    # a in km, angles in rad
    a, e, i, Ω, ω, ν = [float(x) for x in oe]

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
