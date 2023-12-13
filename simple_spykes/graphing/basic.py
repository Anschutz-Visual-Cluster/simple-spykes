from typing import Optional, Union
import matplotlib.pyplot as plt
from simple_spykes.graphing.raw import raw_quality_metrics_unit_graphs, raw_quality_metrics_prob_dists
from simple_spykes.graphing.raw import raw_quality_metrics_correlations
from simple_spykes.graphing.util.grapher import Grapher
from simple_spykes.graphing.util.graphdata import RawGraphData


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
        graph_data.title(f"{save_prefix}{qm_name}")
        if save_folder:
            graph_data.savefig(f"{save_folder}/{save_prefix}unit-{qm_name}.png")
            graph_data.clf()
        else:
            graph_data.show()

        Grapher(graph_data, plt).run()


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
            graph_data.savefig(f"{save_folder}/{save_prefix}prob-{qm_name}.png")
            graph_data.clf()
        else:
            graph_data.show()

        Grapher(graph_data, plt).run()


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
    qm_count = graphing_data[0].get_value("qm_count")

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
        Grapher(graph_data, axes[y_idx, x_idx]).run()

    if save_folder:
        plt.tight_layout()
        plt.savefig(f"{save_folder}/{save_prefix}correlations.png")
        plt.clf()
    else:
        plt.show()


def graph_basic_quality_metrics(metrics_file: Union[str, list[str]], save_folder: Union[bool, str] = False,
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

