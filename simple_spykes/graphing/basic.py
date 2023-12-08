import json
import math
import os
from typing import Optional, Union, Any

import matplotlib.pyplot as plt
import matplotlib.scale
import numpy as np
import sklearn.preprocessing
from scipy.interpolate import UnivariateSpline

GraphingDict = dict[str, dict[str, Any]]
"""
GraphingDict format
{
    # Function to call with x & y values
    "funcname": [x_vals, y_vals],
    
    # When 'funcname' is called, use the params in the dict, optional
    "$funcname$params": {"param1: val, "param2": val2, ...},
    
    # Value to store that could be useful later
    "_value": "Any value here"
}
"""


def _load_file(metrics_file: Union[str, list[str]], exclude: Optional[list[str]] = None, normalize: bool = False,
               use_common_units: bool = False) -> dict:
    if exclude is None:
        exclude = ["epoch_name", "cluster_id", "clusterID", "phy_clusterID"]

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
    if not use_common_units:
        # Check that the data of QMs have the same length, unless using common units then ignore
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
            raise ValueError(
                f"Metrics do not have the same units! '{all_set_keyname}' and '{k}' To ignore set use_common_units=True")
        if use_common_units:
            all_set = all_set.intersection(cur_set)

    ordered_units = sorted(list(all_set))
    new_data = {}
    keylist = list(data.keys())

    for key in keylist:
        to_add = []
        for unit in ordered_units:
            to_add.append(data[key][str(unit)])
        to_add = np.array(to_add)
        if np.all(to_add == None) or np.all(to_add == 0):
            print(f"All values of metric '{key}' are None or 0! Excluding from graphing data")
        else:
            new_data[key] = to_add
    return new_data


def raw_quality_metrics_unit_graphs(metrics_file: str, use_common_units: bool = False) -> GraphingDict:
    """
    Return the calculated values for the graphs, but don't plot them
    :return: GraphingDict data
    """
    data = _load_file(metrics_file, use_common_units=use_common_units)

    quality_metric_names = list(data.keys())
    graphing_data: GraphingDict = {}

    for qm_name in quality_metric_names:
        to_graph = data[qm_name]
        x_vals = [int(v) for v in range(len(to_graph))]
        y_vals = [v or 0 for v in to_graph]
        graphing_data[qm_name] = {"bar": [x_vals, y_vals]}
    return graphing_data


def graph_quality_metrics_unit_graphs(metrics_file: str, save_folder: Union[bool, str] = False,
                                      use_common_units: bool = False, save_prefix: str = ""):
    """
    Shows all the graphs for Unit vs the corresponding quality metric value

    :param metrics_file: string filename of the QM output
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """
    data = _load_file(metrics_file, use_common_units=use_common_units)
    graphing_data = raw_quality_metrics_unit_graphs(metrics_file, use_common_units)

    quality_metric_names = list(data.keys())
    for qm_name in quality_metric_names:
        x_vals, y_vals = graphing_data[qm_name]["bar"]
        plt.bar(x_vals, y_vals)
        plt.title(f"{save_prefix}{qm_name}")
        if save_folder:
            plt.savefig(f"{save_folder}/{save_prefix}unit-{qm_name}.png")
            plt.clf()
        else:
            plt.show()


def raw_quality_metrics_prob_dists(metrics_file: str, use_common_units: bool = False) -> GraphingDict:
    """
    Probability distributions of each metric across the units

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :return: GraphingDict data
    """
    all_data = _load_file(metrics_file, exclude=["epoch_name", "cluster_id"], use_common_units=use_common_units)

    def calc_qm(qm_name: str, qm_data: list[float]) -> Union[bool, GraphingDict]:
        qm_graph_data = {}

        if qm_name in ["epoch_name"]:  # Don't graph these metrics here
            return False
        # TODO remove or set to 0 none vals?
        # qm_values = np.array([v or 0 for v in list(qm_data.values())])
        qm_values = np.array(qm_data)
        qm_values = qm_values[qm_values != None]  # Remove none values

        if len(qm_values) == 0:
            print(f"Can't graph prob dist of metric '{qm_name}' all vals are None")
            return False  # Can't graph this metric

        # bin_size = np.mean(qm_values) / 2
        bin_size = (3.49 * np.std(qm_values) * np.power(len(qm_values), -1 / 3)) / 2
        if bin_size < 0.001:
            bin_size = 0.001
        max_qm_value = np.clip(np.max(qm_values), 0, 9999999)
        min_qm_value = np.clip(np.min(qm_values), -9999999, 0.0001)

        # bin_edges = [bin_size*c for c in range(math.floor(round(max_qm_value/bin_size))]
        num_bins = (max_qm_value - min_qm_value) / bin_size
        bin_edges = np.linspace(
            min_qm_value,
            max_qm_value,
            num=math.ceil(math.fabs(num_bins))
            # Round up to include values that lie in part of a bin_size on the pos edge
        )

        bin_counts = {b: 0 for b in bin_edges}
        for el in qm_values:
            # Find all values in the qm_values that are between the bin edges
            lie_between = bin_edges[(bin_edges - bin_size < el) & (el < bin_edges + bin_size)]
            # Increment the count in the bin of the leftmost edge (-1, last value in the fit)
            bin_counts[lie_between[-1]] = bin_counts[lie_between[-1]] + 1

        bin_counts = np.array(list(bin_counts.values()))
        total = np.sum(bin_counts)
        percentage_weights = bin_counts / total

        qm_graph_data["bar"] = [bin_edges, percentage_weights]
        qm_graph_data["_bin_size"] = bin_size
        qm_graph_data["_bin_count"] = bin_counts
        qm_graph_data["$bar$label"] = f"QM Value with {len(bin_counts)} bins"

        # Fit a spline to the histogram
        spline_func = UnivariateSpline(
            bin_edges - bin_size / 2,  # x vals
            percentage_weights,  # y vals
            s=len(bin_counts)  # smoothing factor
        )

        # Save plot data
        qm_graph_data["plot"] = [bin_edges - bin_size / 2, spline_func(bin_edges - bin_size / 2)]
        qm_graph_data["$plot$color"] = "red"
        qm_graph_data["$plot$linewidth"] = 2
        qm_graph_data["$plot$label"] = "Spline approx"
        qm_graph_data["_binned_by"] = round(bin_size, 2)

        return qm_graph_data

    graphing_data: GraphingDict = {}
    for qm_key_name, qm_value in all_data.items():
        val = calc_qm(qm_key_name, qm_value)
        if val:
            graphing_data[qm_key_name] = val

    return graphing_data


def graph_quality_metrics_prob_dists(metrics_file: str, save_folder: Union[bool, str] = False,
                                     use_common_units: bool = False, save_prefix: str = ""):
    """
    Probability distributions of each metric across the units

    :param metrics_file: string filename to read from
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """
    all_data = _load_file(metrics_file, exclude=["epoch_name", "cluster_id"], use_common_units=use_common_units)

    def graph_qm(qm_name: str, qm_data: dict[str, Any]):
        bin_edges, percentage_weights = qm_data["bar"]
        bin_size = qm_data["_bin_size"]

        plt.bar(
            bin_edges,
            percentage_weights,
            width=bin_size / 2,
            label=qm_data["$bar$label"]
        )

        xspline, yspline = qm_data["plot"]

        # Plot spline
        plt.plot(
            xspline,
            yspline,
            color=qm_data["$plot$color"],
            linewidth=qm_data["$plot$linewidth"],
            label=qm_data["$plot$label"]
        )

        plt.title(f"{save_prefix}{qm_name} Probability Density Histogram")
        plt.xlabel(f"{qm_name} value (binned by {qm_data['_binned_by']})")
        plt.ylabel("Probability")
        plt.legend()
        if save_folder:
            plt.savefig(f"{save_folder}/{save_prefix}prob-{qm_name}.png")
            plt.clf()
        else:
            plt.show()

    graphing_data = raw_quality_metrics_prob_dists(metrics_file, use_common_units)
    for k, v in all_data.items():
        graph_qm(k, graphing_data[k])

# TODO OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

def graph_spikeinterface_quality_metrics_correlations(metrics_file: Union[str, list[str]], save_folder=Optional[str],
                                                      use_common_units: bool = False, save_prefix: str = ""):
    """
    Plot each metric value against each other to determine how they correlate

    :param metrics_file: string filename to read from
    :param save: If set, will save the graph under the given foldername str, else will display
    :return:
    """
    all_data = _load_file(metrics_file, use_common_units=use_common_units, exclude=[
        "l_ratio", "epoch_name",
        "cluster_id", "clusterID", "phy_clusterID", "maxChannels", "nPeaks", "nSpikes", "RPV_tauR_estimate",
        "useTheseTimesStart", "useTheseTimesStop", "nTroughs", "isSomatic", "fractionRPVs_estimatedTauR",
        "ksTest_pValue"
    ])  # Exclude l_ratio since it has no data

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
            s=1
        )
        if x_idx == 0:
            subplot.set_ylabel(y_qm_name, rotation="horizontal", ha="right")
            # subplot.set(ylabel=y_qm_name)
        if y_idx == len(keylist) - 1:
            subplot.set_xlabel(x_qm_name, rotation=90)
            # subplot.set(xlabel=x_qm_name)

        if y_idx != len(keylist) - 1:
            subplot.set_xticks([])
        if x_idx != 0:
            subplot.set_yticks([])

        # print(f"Plotting {x_qm_name}({min(x_data)}, {max(x_data)}) vs {y_qm_name}({min(y_data)}, {max(y_data)})")

    progress = []
    for row in range(qm_count):
        for col in range(qm_count):
            progress.append((row, col))

    fig, axes = plt.subplots(
        nrows=qm_count,
        ncols=qm_count
        # sharex="all",
        # sharey="all",
        # layout="constrained"
    )
    fig.suptitle(f"{save_prefix}Values of QMs against each other")
    fig.set_size_inches(15, 15)
    [plot_subplot(axes, *v) for v in progress]
    if save_folder:
        plt.tight_layout()
        plt.savefig(f"{save_folder}/{save_prefix}correlations.png")
        plt.clf()
    else:
        plt.show()
    tw = 2


def graph_spikeinterface_quality_metrics(metrics_file: Union[str, list[str]], save_folder: Union[bool, str] = False,
                                         use_common_units: bool = False, save_prefix: Optional[str] = ""):
    print(f"Graphing '{metrics_file}'")
    if save_folder:
        if not isinstance(save_folder, str):
            raise ValueError("'save' parameter must be a string if set!")
        # if not os.path.exists(save):
        #     os.mkdir(save)

    # Unit vs qm value
    print("Graphing Units vs Quality Metric Value")
    graph_spikeinterface_quality_metrics_unit_graphs(metrics_file, save_folder=save_folder,
                                                     use_common_units=use_common_units, save_prefix=save_prefix)

    # Probability distribution of the quality metrics values across all units
    print("Graphing Probability Dist of Quality Metric Values")
    graph_spikeinterface_quality_metrics_prob_dists(metrics_file, save_folder=save_folder,
                                                    use_common_units=use_common_units, save_prefix=save_prefix)

    # All quality metrics plotted against another to determine correlations
    print("Graphing Quality Metric v Metric values")
    graph_spikeinterface_quality_metrics_correlations(metrics_file, save_folder=save_folder,
                                                      use_common_units=use_common_units, save_prefix=save_prefix)

    print(f"Done graphing\n--")
    tw = 2
