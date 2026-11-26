import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.8,
    "legend.fontsize": 7.6, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "lines.linewidth": 1.6,
    "mathtext.fontset": "dejavusans", "figure.dpi": 110,
})
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "update_figures")
os.makedirs(FIG, exist_ok=True)


def save(fig, name, dpi=170):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)
