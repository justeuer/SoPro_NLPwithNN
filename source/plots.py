from collections import Counter
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, List


def plot_results(title: str,
                 distances: Dict[str, float],
                 mean_dist: float,
                 mean_dist_norm: float,
                 losses: List[float],
                 outfile: Path):
    fig, (ax_d, ax_l) = plt.subplots(1, 2, figsize=(8, 4))
    # adjust figure size
    fig.tight_layout(pad=3.0)
    plt.suptitle(title)
    plt.figtext(-0.00, -0.00, "\nMean edit distance: {}\nNormalized edit distance: {}"
                .format(round(mean_dist, 2), round(mean_dist_norm, 2)))
    # distances plot
    ax_d.bar(distances.keys(), distances.values())
    ax_d.set_xlabel("Distance")
    ax_d.set_ylabel("Count")
    # loss plot
    l_xs = [str(i) for i in range(1, len(losses)+1)]
    ax_l.plot(l_xs, losses)
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Mean loss")
    # save
    plt.savefig(outfile.absolute())
