from simple_spykes.graphing.basic import graph_spikeinterface_quality_metrics, \
    graph_spikeinterface_quality_metrics_correlations


def graphing(metrics_file, save_prefix):
    graph_spikeinterface_quality_metrics(metrics_file, save_folder="graphs/", save_prefix=save_prefix, use_common_units=True)

    # graph_spikeinterface_quality_metrics(regular_metrics, save_folder="graphs/", save_prefix=save_prefix)
    # graph_spikeinterface_quality_metrics(pc_metrics, save_folder="graphs/", save_prefix=f"{save_prefix}pc-")
    # graph_spikeinterface_quality_metrics_correlations(metrics_file, save_folder="graphs/", save_prefix=f"{save_prefix}all-", use_common_units=True)

    tw = 2


def bombcell():
    from simple_spykes.metric_packages import bombcell_run_quality_metrics
    result = bombcell_run_quality_metrics(
        kilosort_directory="..\\data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP",
        raw_data_directory="..\\data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP\\*.dat",
        metadata_directory="..\\data\\Record Node 105\\experiment1\\recording1\\*.oebin",
        decompress_directory="..\\data\\bombcell_decompress",
        bombcell_save_directory="..\\data\\bombcell_quality_metrics",
        save_filename="bombcell_metrics.json"
    )
    tw = 2


def main():
    # bombcell()

    graphing("spikeinterface_quality_metrics.json", "spikeinterface-")
    graphing("spikeinterface_pc_quality_metrics.json", "spikeinterface-pc-")
    graphing(["spikeinterface_quality_metrics.json", "spikeinterface_pc_quality_metrics.json"], "spikeinterface-all-")
    graphing("bombcell_metrics.json", "bombcell-")
    tw = 2


if __name__ == "__main__":
    main()
