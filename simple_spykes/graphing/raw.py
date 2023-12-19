import math
from typing import Union
import numpy as np
from scipy.interpolate import UnivariateSpline
from simple_spykes.graphing.util.graphdata import RawGraphData, RawGraphVariable
from simple_spykes.graphing.util.io import load_file


def raw_quality_metrics_unit_graphs(metrics_file: Union[str, list[str]], use_common_units: bool = False) -> list[RawGraphData]:
    """
    Return the calculated values for the graphs, but don't plot them

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :return: list of RawGraphData
    """
    data = load_file(metrics_file, use_common_units=use_common_units)

    quality_metric_names = list(data.keys())
    graphing_data: list[RawGraphData] = []

    for qm_name in quality_metric_names:
        to_graph = data[qm_name]
        x_vals = [int(v) for v in range(len(to_graph))]
        y_vals = [v or 0 for v in to_graph]
        graphing_data.append(
            RawGraphData()
            .bar(x_vals, height=y_vals)
            .set_value("qm_name", qm_name)
            .set_value("plot_type", "Unit Graphs")
        )

    return graphing_data


def raw_calc_metric_prob_dist(qm_name: str, qm_data: list[float]) -> Union[bool, RawGraphData]:
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
    bin_size = np.clip(bin_size, 0.0001, 99999999999999999)
    max_qm_value = np.clip(np.max(qm_values), 0, 99999999999999999)
    min_qm_value = np.clip(np.min(qm_values), -99999999999999999, 0.0001)

    # bin_edges = [bin_size*c for c in range(math.floor(round(max_qm_value/bin_size))]
    num_bins = (max_qm_value - min_qm_value) / bin_size
    num_bins = np.clip(num_bins, 1, len(qm_values))
    if num_bins == 1:
        num_bins = len(qm_values)

    bin_edges = np.linspace(
        min_qm_value,
        max_qm_value,
        num=math.ceil(math.fabs(num_bins))
        # Round up to include values that lie in part of a bin_size on the pos edge
    )

    bin_counts_map = {b: 0 for b in bin_edges}
    for el in qm_values:
        if el > max_qm_value:
            bin_counts_map[bin_edges[-1]] = bin_counts_map[bin_edges[-1]] + 1
        elif el < min_qm_value:
            bin_counts_map[bin_edges[0]] = bin_counts_map[bin_edges[0]] + 1
        else:
            # Find all values in the qm_values that are between the bin edges
            lie_between = bin_edges[(bin_edges - bin_size < el) & (el < bin_edges + bin_size)]
            # Increment the count in the bin of the leftmost edge (-1, last value in the fit)
            bin_counts_map[lie_between[-1]] = bin_counts_map[lie_between[-1]] + 1

    bin_counts = np.array(list(bin_counts_map.values()))
    total = np.sum(bin_counts)
    percentage_weights = bin_counts / total

    # Fit a spline to the histogram
    spline_func = UnivariateSpline(
        bin_edges - bin_size / 2,  # x vals
        percentage_weights,  # y vals
        s=len(bin_counts),  # smoothing factor
        k=3 if len(bin_counts) > 3 else 1
    )

    r = RawGraphData() \
        .set_value("bar_label", f"QM Value with {len(bin_counts)} bins") \
        .bar(bin_edges, percentage_weights, width=bin_size/2, label=RawGraphVariable("bar_label")) \
        .set_value("bin_size", bin_size) \
        .set_value("plot_label", "Spline Approx") \
        .plot(bin_edges - bin_size / 2, spline_func(bin_edges - bin_size / 2), linewidth=2, color=RawGraphVariable("spline_color", None), label=RawGraphVariable("plot_label")) \
        .set_value("bin_count", bin_counts) \
        .set_value("binned_by", round(bin_size, 2)) \
        .set_value("plot_type", "Probability Distribution") \
        .xlabel("QM Value") \
        .ylabel("Probability") \
        .set_value("qm_name", qm_name) \
        .set_value("qm_data", qm_data)
    return r


def raw_quality_metrics_prob_dists(metrics_file: Union[str, list[str]], use_common_units: bool = False) -> list[RawGraphData]:
    """
    Probability distributions of each metric across the units

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :return: list of RawGraphData
    """
    all_data = load_file(metrics_file, exclude=["epoch_name", "cluster_id"], use_common_units=use_common_units)

    graphing_data: list[RawGraphData] = []
    for qm_key_name, qm_value in all_data.items():
        val = raw_calc_metric_prob_dist(qm_key_name, qm_value)
        if val:
            val.set_value("qm_name", qm_key_name)
            graphing_data.append(val)

    return graphing_data


def raw_quality_metrics_correlations(metrics_file: Union[str, list[str]], use_common_units: bool = False) -> list[RawGraphData]:
    """
    Plot each metric value against each other to determine how they correlate

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that are present in all QM results
    :return: list of RawGraphData
    """

    all_data = load_file(metrics_file, use_common_units=use_common_units, exclude=[
        "epoch_name", "cluster_id", "clusterID", "phy_clusterID", "maxChannels", "nPeaks", "nSpikes",
        "RPV_tauR_estimate", "useTheseTimesStart", "useTheseTimesStop", "nTroughs", "isSomatic",
        "fractionRPVs_estimatedTauR", "ksTest_pValue"
    ])  # Exclude

    qm_count = len(list(all_data.keys()))

    def raw_subplot(x_idx, y_idx) -> RawGraphData:
        graph_data = RawGraphData() \
            .set_value("y_idx", y_idx) \
            .set_value("x_idx", x_idx) \
            .set_value("plot_type", "Correlations") \
            .set_value("qm_count", qm_count)

        keylist = list(all_data.keys())

        x_qm_name = keylist[x_idx]
        x_data = all_data[x_qm_name]

        y_qm_name = keylist[y_idx]
        y_data = all_data[y_qm_name]

        graph_data.scatter(x=x_data, y=y_data, s=1)

        if x_idx == 0:
            graph_data.add_func("set_ylabel", [], {"ylabel": y_qm_name, "rotation": "horizontal", "ha": "right"})
        if y_idx == len(keylist) - 1:
            graph_data.add_func("set_xlabel", [], {"xlabel": x_qm_name, "rotation": 90})

        if y_idx != len(keylist) - 1:
            graph_data.add_func("set_xticks", [], {"ticks": []})
        if x_idx != 0:
            graph_data.add_func("set_yticks", [], {"ticks": []})
        return graph_data

    progress = []
    for row in range(qm_count):
        for col in range(qm_count):
            progress.append((col, row))

    return [raw_subplot(*v) for v in progress]

