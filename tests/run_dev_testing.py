# from simple_spykes.util.ecephys import run_quality_metrics
from simple_spykes.util.spikeinterface_util import run_quality_metrics
from simple_spykes.graphing.basic import graph_spikeinterface_quality_metrics, \
    graph_spikeinterface_quality_metrics_correlations

def graphing():
    regular_metrics = "quality_metrics.json"
    pc_metrics = "pc_quality_metrics.json"

    all_metrics = ["quality_metrics.json", "pc_quality_metrics.json"]

    metrics_file = regular_metrics
    # metrics_file = pc_metrics
    # metrics_file = all_metrics

    # use_common_units = True

    # graph_spikeinterface_quality_metrics(metrics_file, use_common_units=True)
    graph_spikeinterface_quality_metrics(regular_metrics, save_folder="graphs/")
    graph_spikeinterface_quality_metrics(pc_metrics, save_folder="graphs/", save_prefix="pc-")
    graph_spikeinterface_quality_metrics_correlations(metrics_file, save_folder="graphs/", save_prefix="all-", use_common_units=True)

    tw = 2


def main():
    # Data from \\10.33.107.246\s2\stim\recordings\jlh34_2023-05-15_16-03-21\Record Node 105\experiment1\recording1\continuous\Neuropix-PXI-104.ProbeA-AP

    r = run_quality_metrics(
        folder_path="../data\\Record Node 105",
        stream_name="Neuropix-PXI-104.ProbeA-AP",
        kilosort_output_directory="../data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP"
    )


if __name__ == "__main__":
    main()
