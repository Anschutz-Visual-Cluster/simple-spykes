# from simple_spykes.util.ecephys import run_quality_metrics
from simple_spykes.util.spikeinterface_util import run_quality_metrics
from simple_spykes.graphing.basic import graph_spikeinterface_quality_metrics

def main():
    # Yoinked from \\10.33.107.246\s2\stim\recordings\jlh34_2023-05-15_16-03-21\Record Node 105\experiment1\recording1\continuous\Neuropix-PXI-104.ProbeA-AP

    # r = run_quality_metrics(
    #     folder_path="../data\\Record Node 105",
    #     stream_name="Neuropix-PXI-104.ProbeA-AP",
    #     kilosort_output_directory="../data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP"
    # )

    # metrics_file = "quality_metrics.json"
    # metrics_file = "pc_quality_metrics.json"
    metrics_file = ["quality_metrics.json", "pc_quality_metrics.json"]

    # save = "graphs/"
    save = False

    graph_spikeinterface_quality_metrics(metrics_file, save=save)

    tw = 2


if __name__ == "__main__":
    main()
