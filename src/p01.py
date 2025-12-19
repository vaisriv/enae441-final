# libs
import matplotlib.pyplot as plt
import numpy as np

# src
from measurements import *

def e(data: measurements.Measurement) -> plt.Figure:
    t = np.asarray(data.t)
    i = np.asarray(data.i).astype(int)
    ρ = np.asarray(data.ρ)

    labels = {0: "DSN #0 Goldstone", 1: "DSN #1 Madrid", 2: "DSN #2 Canberra"}
    colors = {0: "r", 1: "g", 2: "b"}

    order = np.argsort(t)
    t, i, ρ = t[order], i[order], ρ[order]

    fig, ax = plt.subplots()

    for stn in np.unique(i):
        mask = (i == stn)
        ax.plot(t[mask], ρ[mask],
                linestyle="-", linewidth=0,
                marker=".", markersize=2,
                color=colors.get(int(stn)),
                label=labels.get(int(stn), f"DSN #{int(stn)}"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc = "upper left")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Range [km]")
    ax.set_title("DSN Range [km] vs. Time [s]")

    fig.tight_layout()

    return fig
