import time
from typing import Union, Optional, Callable

from simple_spykes.graphing.raw import raw_quality_metrics_prob_dists, raw_calc_metric_prob_dist
from simple_spykes.graphing.util.graphdata import RawGraphData
from simple_spykes.graphing.util.grapher import Grapher
import matplotlib.pyplot as plt
from matplotlib import colormaps
from simple_spykes.graphing.util.io import load_file


def load_multi_data(metric_files_list: list[list[str]], exclude: list[str] = list()):
    all_loaded_data = []
    qm_count = None
    qm_names = None
    common_units = None

    # Load files
    for metric_files in metric_files_list:
        if not isinstance(metric_files, list):
            raise ValueError(f"metric_files_list must be a list of lists! Found type {type(metric_files)} instead! Value: {metric_files}")

        loaded_datas = []
        # Load each metric file
        for metric_file in metric_files:
            loaded_file_data = load_file(metric_file, use_common_units=True, exclude=exclude)
            loaded_datas.append(loaded_file_data)

            # Check that qm_count is the same between datasets TODO Common QMs only?
            if qm_count is None:
                qm_count = len(loaded_file_data.keys())
                qm_names = set(loaded_file_data.keys())

            cur_names = set(loaded_file_data.keys())
            if qm_names != cur_names:
                raise ValueError(f"Previous metric names do not match up with metric names in file: '{metric_file}' Old set: {qm_names} New set: {cur_names} Diff: {qm_names.difference(cur_names)}")

        all_loaded_data.append(loaded_datas)

    # Aggregate across files by qm
    all_raw_data = []
    for overlay_datas in all_loaded_data:
        aggregated_data = [[] for _ in range(qm_count)]
        for metric_file_data in overlay_datas:
            for qm_idx, qm_name in enumerate(qm_names):
                aggregated_data[qm_idx].extend(metric_file_data[qm_name])
        all_raw_data.append(aggregated_data)
    return qm_count, qm_names, all_raw_data


def raw_multi_graph(metric_files_list: list[list[str]],
                    single_qm_func: Callable[[str, list[float]], Union[RawGraphData, bool]],
                    modify_subgraph: Callable[[RawGraphData, str], RawGraphData],
                    title_format_str: str,
                    exclude: list[str] = list(),
                    labels: Optional[list[str]] = None) -> list[RawGraphData]:
    """
    Raw data for multiple graphs

    :param metric_files_list: list of list of string filenames to read from. Will aggregate the inner lists together
    :param single_qm_func: Function returning a list of graphs corresponding to the quality metrics for a given list of
     floats and the qm name
    :param modify_subgraph: Function to run on each subgraph eg [[subgraph01, subgraph02, ...], [subgraph11, subgraph12, ..], ..]
     when combining subgraph01, subgraph11, ... into a single graph. Function like f(graph, label) -> graph where label
     is the corresponding label from the labels arg
    :param title_format_str: Formattable string for the title. Must include {qm_name} somewhere
    :param exclude: List of string of qm names to exclude from loading
    :param labels: List of labels for each set of aggregated data. eg ['raw', 'curated'] If not set will default to idx
    :return: list of finished RawGraphData
    """

    if labels is None:
        labels = [str(i) for i in range(len(metric_files_list))]

    if len(labels) != len(metric_files_list):
        raise ValueError("Length of labels != Length of the metric files list!")

    if not isinstance(metric_files_list, list):
        raise ValueError("metric_files_list must be a list!")

    qm_count, qm_names, all_raw_data = load_multi_data(metric_files_list=metric_files_list, exclude=exclude)

    # Process into graphs
    final_graphs = [RawGraphData() for _ in range(qm_count)]
    for to_aggregate_idx, aggregated_datas in enumerate(all_raw_data):

        # Make new graphs with the aggregated data
        for qm_idx, qm_name in enumerate(qm_names):
            g = single_qm_func(qm_name, aggregated_datas[qm_idx])
            if not g:  # Ignore results that don't have any data TODO?
                continue

            g.title(title_format_str.format(qm_name=qm_name))
            label = labels[to_aggregate_idx]

            # Run modifications on individual graphs
            g = modify_subgraph(g, label)
            # Lock in vars so they don't get modified by appending graphs
            g.replace_vars()

            final_graphs[qm_idx].append_graph(g)

    return final_graphs


def graph_multi_graph(metric_files_list: list[list[str]],
                      single_qm_func: Callable[[str, list[float]], Union[RawGraphData, bool]],
                      modify_subgraph: Callable[[RawGraphData, str], RawGraphData],
                      title_format_str: str,
                      exclude: list[str] = list(),
                      labels: Optional[list[str]] = None,
                      save_folder: Union[bool, str] = False,
                      save_prefix: str = ""):
    """
    Plots multiple metric files

    :param metric_files_list: list of list of string filenames to read from. Will aggregate the inner lists together
    :param single_qm_func: Function returning a list of graphs corresponding to the quality metrics for a given list of
     floats and the qm name
    :param modify_subgraph: Function to run on each subgraph eg [[subgraph01, subgraph02, ...], [subgraph11, subgraph12, ..], ..]
     when combining subgraph01, subgraph11, ... into a single graph. Function like f(graph, label) -> graph where label
     is the corresponding label from the labels arg
    :param title_format_str: Formattable string for the title. Must include {qm_name} somewhere
    :param exclude: List of string of qm names to exclude from loading
    :param labels: List of labels for each set of aggregated data. eg ['raw', 'curated'] If not set will default to idx
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """
    final_graphs = raw_multi_graph(
        metric_files_list=metric_files_list,
        single_qm_func=single_qm_func,
        modify_subgraph=modify_subgraph,
        title_format_str=title_format_str,
        exclude=exclude,
        labels=labels
    )

    # Plot and show the graphs
    for g in final_graphs:
        g.legend()

        if save_folder:
            g.savefig(f"{save_folder}/{save_prefix}multiprob-{g.get_value('qm_name')}.png")
            g.clf()
        else:
            g.show()

        Grapher(g, plt).run()


def graph_multi_prob_dists(metric_files_list: list[list[str]], labels: Optional[list[str]] = None,
                           save_folder: Union[bool, str] = False, save_prefix: str = ""):
    """
    Probability distributions of each metric across the units, for multiple metric files

    :param metric_files_list: list of list of string filenames to read from. Will aggregate the inner lists together
    :param labels: List of labels for each set of aggregated data. eg ['raw', 'curated'] If not set will default to idx
    :param save_folder: If set, will save the graph in the folder of the string value given, else False will only show plots
    :param save_prefix: prefix to put in front of the saved file, not the same as the directory
    :return: None
    """

    def mod_graph(g: RawGraphData, label: str) -> RawGraphData:
        g.set_value("bar_label", f"{label} {g.get_value('bar_label')}")
        # g.set_value("spline_color",  None)  # colormaps.get("Dark2").colors[labels.index(label)]
        return g

    graph_multi_graph(
        metric_files_list=metric_files_list,
        single_qm_func=raw_calc_metric_prob_dist,
        modify_subgraph=mod_graph,
        title_format_str="{qm_name} Multi Probability Density Histogram",
        exclude=[
            "epoch_name", "cluster_id", "clusterID", "phy_clusterID", "maxChannels", "nPeaks", "nSpikes",
            "RPV_tauR_estimate", "useTheseTimesStart", "useTheseTimesStop", "nTroughs", "isSomatic",
            "fractionRPVs_estimatedTauR", "ksTest_pValue"],
        labels=labels,
        save_folder=save_folder,
        save_prefix=save_prefix
    )
