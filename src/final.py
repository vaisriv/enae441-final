# libs
import numpy as np
import matplotlib.pyplot as plt
import os

# src
from measurements import *

def main() -> int:
    data = numpy_data.load("./data/Project-Measurements-Easy.npy")

    #######
    # p01e #
    #######
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

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Range [km]")
    ax.set_title("DSN Range [km] vs. Time [s]")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "./outputs/figures/s01e.png")
    plt.savefig(save_path)
    # plt.show()

    return 0


if __name__ == "__main__":
    main()
