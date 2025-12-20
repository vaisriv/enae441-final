# libs
import matplotlib.pyplot as plt
import numpy as np

# src
from measurements import *

def e(data: measurements.Measurement) -> plt.Figure:
    t = np.asarray(data.t)
    i = np.asarray(data.i).astype(int)
    ρ = np.asarray(data.ρ)
    dρ = np.asarray(data.dρ)

    labels = {0: "DSN #0 Goldstone", 1: "DSN #1 Madrid", 2: "DSN #2 Canberra"}
    colors = {0: "r", 1: "g", 2: "b"}

    order = np.argsort(t)
    t, i, ρ, dρ = t[order], i[order], ρ[order], dρ[order]

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(12, 8))

    for stn in np.unique(i):
        mask = (i == stn)
        axs[0].plot(t[mask], ρ[mask],
                linestyle="-", linewidth=0,
                marker=".", markersize=2,
                color=colors.get(int(stn)),
                label=labels.get(int(stn), f"DSN #{int(stn)}"))
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc = "upper left")
        axs[0].set_ylabel("Range [km]")

    for stn in np.unique(i):
        mask = (i == stn)
        axs[1].plot(t[mask], dρ[mask],
                linestyle="-", linewidth=0,
                marker=".", markersize=2,
                color=colors.get(int(stn)),
                label=labels.get(int(stn), f"DSN #{int(stn)}"))
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc = "upper left")
        axs[1].set_ylabel("Range-Rate [km/s]")

    fig.supxlabel("Time [s]")
    fig.suptitle("DSN Measurements vs. Time [s]")
    fig.tight_layout()

    return fig
