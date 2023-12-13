import os
from simple_spykes.graphing.basic import graph_quality_metrics


def graphing(metrics_file, save_prefix, do_save):
    save_folder = "mass_graphs/"
    if not do_save:
        save_folder = False

    graph_quality_metrics(metrics_file, save_folder=save_folder, save_prefix=save_prefix, use_common_units=True)


def graphing_testing():
    do_save = True

    graphing("bombcell-2023-04-11-raw.json", "bombcell-2023-04-11-raw-", do_save)
    tw = 2


if __name__ == "__main__":
    graphing_testing()

