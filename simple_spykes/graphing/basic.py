import json
import math
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from simple_spykes.graphing.grapher import Grapher
from simple_spykes.graphing.raw_graphdata import RawGraphData


def _load_file(metrics_file: Union[str, list[str]], exclude: Optional[list[str]] = None,
               use_common_units: bool = False) -> dict:
    """
    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param exclude: List of string values from the qm file to exclude from loading
    :return: dict of QM formatted metric_data
    """
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
    #     for k, v in metric_data.items():
    #         d2[k] = {c: sklearn.preprocessing.normalize([list(v.values())]) for c in range(len(v.keys()))}
    #         tw = 2
    #     metric_data = d2
    #
    if not use_common_units:
        # Check that the metric_data of QMs have the same length, unless using common units then ignore
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
            print(f"All values of metric '{key}' are None or 0! Excluding from graphing metric_data")
        else:
            new_data[key] = to_add
    return new_data


def raw_quality_metrics_unit_graphs(metrics_file: Union[str, list[str]], use_common_units: bool = False) -> list[RawGraphData]:
    """
    Return the calculated values for the graphs, but don't plot them

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :return: list of RawGraphData
    """
    data = _load_file(metrics_file, use_common_units=use_common_units)

    quality_metric_names = list(data.keys())
    graphing_data: list[RawGraphData] = []

    for qm_name in quality_metric_names:
        to_graph = data[qm_name]
        x_vals = [int(v) for v in range(len(to_graph))]
        y_vals = [v or 0 for v in to_graph]
        graphing_data.append(
            RawGraphData()
            .add_func("bar", {"x": x_vals, "height": y_vals})
            .add_value("qm_name", qm_name)
            .add_value("plot_type", "Unit Graphs")
        )

    return graphing_data


def graph_quality_metrics_unit_graphs(metrics_file: Union[str, list[str]], save_folder: Union[bool, str] = False,
                                      use_common_units: bool = False, save_prefix: str = ""):
    """
    Shows all the graphs for Unit vs the corresponding quality metric value

    :param metrics_file: string filename of the QM output
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """
    graphing_data: list[RawGraphData] = raw_quality_metrics_unit_graphs(metrics_file, use_common_units)

    for graph_data in graphing_data:
        qm_name = graph_data.get_value("qm_name")
        graph_data.add_func("title", [f"{save_prefix}{qm_name}"])
        if save_folder:
            graph_data.add_func("savefig", [f"{save_folder}/{save_prefix}unit-{qm_name}.png"])
            graph_data.clf()
        else:
            graph_data.show()

        Grapher(graph_data, plt).run()


def raw_quality_metrics_prob_dists(metrics_file: Union[str, list[str]], use_common_units: bool = False) -> list[RawGraphData]:
    """
    Probability distributions of each metric across the units

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :return: list of RawGraphData
    """
    all_data = _load_file(metrics_file, exclude=["epoch_name", "cluster_id"], use_common_units=use_common_units)

    def calc_qm(qm_name: str, qm_data: list[float]) -> Union[bool, RawGraphData]:
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

        # Fit a spline to the histogram
        spline_func = UnivariateSpline(
            bin_edges - bin_size / 2,  # x vals
            percentage_weights,  # y vals
            s=len(bin_counts)  # smoothing factor
        )

        return RawGraphData() \
            .add_func(
            "bar", {
                "x": bin_edges,
                "height": percentage_weights,
                "label": f"QM Value with {len(bin_counts)} bins"
            }) \
            .add_value("bin_size", bin_size) \
            .add_value("bin_count", "bin_counts") \
            .add_func("plot",[
                bin_edges - bin_size / 2,
                spline_func(bin_edges - bin_size / 2),
            ],
            {
                "color": "red",
                "linewidth": 2,
                "label": "Spline Approx"
            }) \
            .add_value("binned_by", round(bin_size, 2)) \
            .add_value("plot_type", "Probability Distribution")

    graphing_data: list[RawGraphData] = []
    for qm_key_name, qm_value in all_data.items():
        val = calc_qm(qm_key_name, qm_value)
        if val:
            val.add_value("qm_name", qm_key_name)
            graphing_data.append(val)

    return graphing_data


def graph_quality_metrics_prob_dists(metrics_file: Union[str, list[str]], save_folder: Union[bool, str] = False,
                                     use_common_units: bool = False, save_prefix: str = ""):
    """
    Probability distributions of each metric across the units

    :param metrics_file: string filename to read from
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """

    graphing_data: list[RawGraphData] = raw_quality_metrics_prob_dists(metrics_file, use_common_units)
    for graph_data in graphing_data:
        qm_name = graph_data.get_value("qm_name")
        binned_by = graph_data.get_value("binned_by")

        graph_data.add_func("title", [f"{save_prefix}{qm_name} Probability Density Histogram"])
        graph_data.add_func("xlabel", [f"{qm_name} value (binned by {binned_by})"])
        graph_data.add_func("ylabel", ["Probability"])
        graph_data.simple("legend")
        if save_folder:
            graph_data.add_func("savefig", [f"{save_folder}/{save_prefix}prob-{qm_name}.png"])
            graph_data.clf()
        else:
            graph_data.show()

        Grapher(graph_data, plt).run()


def raw_quality_metrics_correlations(metrics_file: Union[str, list[str]], use_common_units: bool = False) -> list[RawGraphData]:
    """
    Plot each metric value against each other to determine how they correlate

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that are present in all QM results
    :return: list of RawGraphData
    """

    all_data = _load_file(metrics_file, use_common_units=use_common_units, exclude=[
        "epoch_name", "cluster_id", "clusterID", "phy_clusterID", "maxChannels", "nPeaks", "nSpikes",
        "RPV_tauR_estimate", "useTheseTimesStart", "useTheseTimesStop", "nTroughs", "isSomatic",
        "fractionRPVs_estimatedTauR", "ksTest_pValue"
    ])  # Exclude

    qm_count = len(list(all_data.keys()))

    def raw_subplot(x_idx, y_idx) -> RawGraphData:
        graph_data = RawGraphData() \
            .add_value("y_idx", y_idx) \
            .add_value("x_idx", x_idx) \
            .add_value("plot_type", "Correlations")

        keylist = list(all_data.keys())

        x_qm_name = keylist[x_idx]
        x_data = all_data[x_qm_name]

        y_qm_name = keylist[y_idx]
        y_data = all_data[y_qm_name]

        graph_data.add_func(
            "scatter",
            {"x": x_data, "y": y_data, "s": 1}
        )

        if x_idx == 0:
            graph_data.add_func("set_ylabel", {"ylabel": y_qm_name, "rotation": "horizontal", "ha": "right"})
        if y_idx == len(keylist) - 1:
            graph_data.add_func("set_xlabel", {"xlabel": x_qm_name, "rotation": 90})

        if y_idx != len(keylist) - 1:
            graph_data.add_func("set_xticks", {"ticks": []})
        if x_idx != 0:
            graph_data.add_func("set_yticks", {"ticks": []})
        return graph_data

    progress = []
    for row in range(qm_count):
        for col in range(qm_count):
            progress.append((col, row))

    return [raw_subplot(*v) for v in progress]


def graph_quality_metrics_correlations(metrics_file: Union[str, list[str]], save_folder=Optional[str],
                                       use_common_units: bool = False, save_prefix: str = ""):
    """
    Plot each metric value against each other to determine how they correlate

    :param metrics_file: string filename to read from
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """
    graphing_data = raw_quality_metrics_correlations(metrics_file, use_common_units)
    qm_count = len(graphing_data)

    progress = []
    for row in range(qm_count):
        for col in range(qm_count):
            progress.append((col, row))

    fig, axes = plt.subplots(
        nrows=qm_count,
        ncols=qm_count
    )

    fig.suptitle(f"{save_prefix}Values of QMs against each other")
    fig.set_size_inches(15, 15)

    for graph_data in graphing_data:
        x_idx = graph_data.get_value("x_idx")
        y_idx = graph_data.get_value("y_idx")
        Grapher(graph_data, axes[x_idx, y_idx]).run()

    if save_folder:
        plt.tight_layout()
        plt.savefig(f"{save_folder}/{save_prefix}correlations.png")
        plt.clf()
    else:
        plt.show()


def graph_quality_metrics(metrics_file: Union[str, list[str]], save_folder: Union[bool, str] = False,
                          use_common_units: bool = False, save_prefix: Optional[str] = ""):
    """
    Plot all basic quality metrics
    - Unit Graph
    - Probability Distributions
    - Metric Correlations

    :param metrics_file: string filename to read from
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """

    print(f"Graphing '{metrics_file}'")
    if save_folder:
        if not isinstance(save_folder, str):
            raise ValueError("'save' parameter must be a string if set!")
        # if not os.path.exists(save):
        #     os.mkdir(save)

    # Unit vs qm value
    print("Graphing Units vs Quality Metric Value")
    graph_quality_metrics_unit_graphs(metrics_file, save_folder=save_folder,
                                                     use_common_units=use_common_units, save_prefix=save_prefix)

    # Probability distribution of the quality metrics values across all units
    print("Graphing Probability Dist of Quality Metric Values")
    graph_quality_metrics_prob_dists(metrics_file, save_folder=save_folder,
                                                    use_common_units=use_common_units, save_prefix=save_prefix)

    # All quality metrics plotted against another to determine correlations
    print("Graphing Quality Metric v Metric values")
    graph_quality_metrics_correlations(metrics_file, save_folder=save_folder,
                                                      use_common_units=use_common_units, save_prefix=save_prefix)

    print(f"Done graphing")
    print("--")

