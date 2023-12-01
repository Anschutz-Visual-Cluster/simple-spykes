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


def bombcell():
    from simple_spykes.metric_packages import bombcell_run_quality_metrics
    bombcell_run_quality_metrics(
        kilosort_directory="..\\data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP",
        raw_data_directory="..\\data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP\\*.dat",
        metadata_directory="..\\data\\Record Node 105\\experiment1\\recording1\\*.oebin",
        decompress_directory="..\\data\\bombcell_decompress",
        bombcell_save_directory="..\\data\\bombcell_quality_metrics",
        save_filename="bombcell_metrics.json"
    )


def main():
    bombcell()
    tw = 2


if __name__ == "__main__":
    main()
