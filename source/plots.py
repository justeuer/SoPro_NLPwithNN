from collections import Counter
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, List


def plot_results(title: str,
                 distances: Dict[str, float],
                 percentiles: Dict[str, float],
                 mean_dist: float,
                 mean_dist_norm: float,
                 losses: List[float],
                 outfile: Path):
    fig, (ax_d, ax_p, ax_l) = plt.subplots(1, 3, figsize=(10, 4))

    # adjust figure size
    fig.tight_layout(pad=3.0)
    plt.suptitle(title)
    plt.figtext(-0.00, -0.00, "\nMean edit distance: {}\nNormalized edit distance: {}"
                .format(round(mean_dist, 2), round(mean_dist_norm, 2)))

    # distances plot
    bars = ax_d.bar(distances.keys(), distances.values())
    # add labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax_d.annotate(
            "{}".format(height),
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom'
        )
    ax_d.title.set_text("a) Distances")
    ax_d.set_xlabel("Distance")
    ax_d.set_ylabel("Count")

    # percentiles plot
    x = list(percentiles.keys())[1:]
    y = list(percentiles.values())[1:]
    bars = ax_p.barh(x, y)
    # add labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        indent = 0.2 if i == len(y) else 0.05
        #indent = 0.05
        #if i == 0:
        #    indent = -0.05
        ax_p.annotate(
            "{}%".format(int(width*100)),
            xy=(indent + bar.get_width(), bar.get_y()+0.3*bar.get_height()),
            ha='center',
            va='bottom'
        )
    ax_p.title.set_text("b) Percentiles")
    ax_p.set_xlabel("Percentile")
    ax_p.set_ylabel("Distance")

    # loss plot
    l_xs = [str(i) for i in range(1, len(losses)+1)]
    ax_l.plot(l_xs, losses)
    ax_l.title.set_text("c) Training loss")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Mean loss")
    # save
    plt.savefig(outfile.absolute())
