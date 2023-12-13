import os
from simple_spykes.graphing.basic import graph_quality_metrics


def graphing(metrics_file, save_prefix, do_save):
    save_folder = "graphs/"
    if not do_save:
        save_folder = False

    graph_quality_metrics(metrics_file, save_folder=save_folder, save_prefix=save_prefix, use_common_units=True)


def graphing_testing():
    os.chdir("metric_data")

    do_save = False

    graphing("spikeinterface_quality_metrics.json", "spikeinterface-", do_save)
    graphing("spikeinterface_pc_quality_metrics.json", "spikeinterface-pc-", do_save)
    graphing(["spikeinterface_quality_metrics.json", "spikeinterface_pc_quality_metrics.json"], "spikeinterface-all-",
             do_save)
    graphing("bombcell_metrics.json", "bombcell-", do_save)
    tw = 2


if __name__ == "__main__":
    graphing_testing()

