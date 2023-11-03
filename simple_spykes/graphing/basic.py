import json
import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.scale
import numpy as np
import sklearn.preprocessing
from scipy.interpolate import UnivariateSpline


def _load_file(metrics_file: Union[str, list[str]], exclude: Optional[list[str]] = None, normalize:bool = False, use_common_units: bool = False) -> dict:
    if exclude is None:
        exclude = ["epoch_name", "cluster_id"]

    if not isinstance(metrics_file, list):
        metrics_file = [metrics_file]

    data = {}
    for filename in metrics_file:
        with open(filename, "r") as f:
            loaddata = json.load(f)
        [loaddata.pop(ex, None) for ex in exclude]
        data.update(loaddata)

    # Normalize broken rn
    # if normalize:
    #     d2 = {}
    #     for k, v in data.items():
    #         d2[k] = {c: sklearn.preprocessing.normalize([list(v.values())]) for c in range(len(v.keys()))}
    #         tw = 2
    #     data = d2
    #
    val_len = None
    key_used_for_len = None
    for k, v in data.items():
        if val_len is None:
            val_len = len(v)
            key_used_for_len = k
        if len(v) != val_len:
            raise ValueError(f"Error, length of QM '{k}' isn't the same size as '{key_used_for_len}'")

    new_data = {}

    # Ensure common units align with each other

    all_set = None
    all_set_keyname = None
    for k, v in data.items():  # Keeping separate from above for loop so it can be easily commented out
        if all_set is None:
            all_set = set([int(s) for s in v.keys()])
            all_set_keyname = k
        cur_set = set([int(s) for s in v.keys()])
        if cur_set != all_set and not use_common_units:
            raise ValueError(f"Metrics do not have the same units! '{all_set_keyname}' and '{k}' To ignore set use_common_units=True")
        if use_common_units:
            all_set = all_set.intersection(cur_set)

    ordered_units = sorted(list(all_set))
    new_data = {}
    keylist = list(data.keys())

    for key in keylist:
        new_data[key] = []
        for unit in ordered_units:
            new_data[key].append(data[key][str(unit)])
    tw = 2

    return new_data


def graph_spikeinterface_quality_metrics_unit_graphs(metrics_file: str, save: Union[bool, str] = False):
    """
    Shows all the graphs for Unit vs the corresponding quality metric value

    :param metrics_file: string filename of the QM output
    :param save: If set, will save the graph in the folder of the string value given, else False will only show plots
    :return:
    """
    data = _load_file(metrics_file)

    quality_metric_names = list(data.keys())
    for qm_name in quality_metric_names:
        to_graph = data[qm_name]
        x_vals = [int(v) for v in list(to_graph.keys())]
        y_vals = [v or 0 for v in list(to_graph.values())]

        plt.bar(x_vals, y_vals)
        plt.title(qm_name)
        if save:
            plt.savefig(f"{save}{qm_name}-unit.png")
            plt.clf()
        else:
            plt.show()


def graph_spikeinterface_quality_metrics_prob_dists(metrics_file: str, save: Union[bool, str] = False):
    """
    Probability distributions of each metric across the units

    :param metrics_file: string filename to read from
    :param save: If set, will save the graph in the folder of the string value given, else False will only show plots
    :return:
    """
    all_data = _load_file(metrics_file, exclude=["epoch_name", "cluster_id"])

    def graph_qm(qm_name: str, qm_data: dict):
        if qm_name in ["epoch_name"]:  # Can't graph these metrics here
            return

        # TODO remove or set to 0 none vals?
        # qm_values = np.array([v or 0 for v in list(qm_data.values())])
        qm_values = np.array(list(qm_data.values()))
        qm_values = qm_values[qm_values != None]  # Remove none values

        if len(qm_values) == 0:
            print(f"Can't graph prob dist of metric '{qm_name}' all vals are None")
            return  # Can't graph this metric

        # manual_bins = np.arange(0, len(qm_values))
        # bins = 1

        bins = len(qm_values)//10


        # smallest_step = np.abs(np.diff(qm_values)[np.diff(qm_values) != 0]).min()
        # manual_bins = np.arange(qm_values.min(), qm_values.max())


        hist, bin_edges = np.histogram(qm_values, bins=bins, density=True)
        # hist, bin_edges = np.histogram(qm_values, bins=manual_bins, density=True)
        bin_edge_half_size = (bin_edges[1] - bin_edges[0]) / 2
        bin_centers = bin_edges[:-1] + bin_edge_half_size  # Offset edges by half to center values

        # Graph histogram of values
        plt.bar(
            [c for c in range(len(hist))],
            hist,
            label=f"QM Value with bin size {bins}"
        )
        # Fit a spline to the histogram
        spline_func = UnivariateSpline(
            bin_centers,  # x vals
            hist,  # y vals
            s=bins  # smoothing factor
        )
        # Plot spline
        plt.plot(
            [c for c in range(len(bin_centers))],
            spline_func(bin_centers),
            color="red",
            linewidth=2,
            label="Spline approx"
        )
        plt.title(f"{qm_name} Probability Density Histogram")
        plt.xlabel(f"{qm_name} value (binned)")
        plt.ylabel("Probability")
        plt.legend()
        if save:
            plt.savefig(f"{save}{qm_name}-prob.png")
            plt.clf()
        else:
            plt.show()
        tw = 2
        pass

    for k, v in all_data.items():
        graph_qm(k, v)


def graph_spikeinterface_quality_metrics_correlations(metrics_file: Union[str, list[str]]):
    """
    Plot each metric value against each other to determine how they correlate

    :param metrics_file: string filename to read from
    :return:
    """
    all_data = _load_file(metrics_file) #, normalize=True)

    qm_count = len(list(all_data.keys()))

    def plot_subplot(ax, x_idx, y_idx):
        subplot = ax[y_idx, x_idx]  # Flip plot index to align axes

        keylist = list(all_data.keys())

        x_qm_name = keylist[x_idx]
        x_data = all_data[x_qm_name]

        y_qm_name = keylist[y_idx]
        y_data = all_data[y_qm_name]

        subplot.scatter(
            x=x_data,
            y=y_data,
            marker=","
        )
        if x_idx == 0:
            subplot.set(ylabel=y_qm_name)
        if y_idx == len(keylist)-1:
            subplot.set(xlabel=x_qm_name)

        print(f"Plotting {x_qm_name}({min(x_data)}, {max(x_data)}) vs {y_qm_name}({min(y_data)}, {max(y_data)})")

        # subplot.set_xlim(0, max(x_data))
        # subplot.set_ylim(0, max(y_data))

    progress = []
    for row in range(qm_count):
        for col in range(qm_count):
            progress.append((row, col))
            _, axes = plt.subplots(
                nrows=qm_count,
                ncols=qm_count,
                s=10
                # sharex="all",
                # sharey="all",
                # layout="constrained"
            )
            [plot_subplot(axes, *v) for v in progress]
            plt.show()
            plt.close()
            tw = 2
    tw = 2
    pass


def graph_spikeinterface_quality_metrics(metrics_file: Union[str, list[str]], save: Union[bool, str] = False):
    if save:
        if not isinstance(save, str):
            raise ValueError("'save' parameter must be a string if set!")
        if not os.path.exists(save):
            os.mkdir(save)

    # Unit vs qm value
    # graph_spikeinterface_quality_metrics_unit_graphs(metrics_file, save=save)

    # Probability distribution of the quality metrics values across all units
    # graph_spikeinterface_quality_metrics_prob_dists(metrics_file, save=save)

    # All quality metrics plotted against another to determine correlations
    graph_spikeinterface_quality_metrics_correlations(metrics_file)

    tw = 2
