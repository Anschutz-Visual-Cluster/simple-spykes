import os
from simple_spykes.metric_packages import bombcell_run_quality_metrics


def bombcell_testing():
    os.chdir("metric_data")
    result = bombcell_run_quality_metrics(
        kilosort_directory="..\\..\\data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP",
        raw_data_directory="..\\..\\data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP\\*.dat",
        metadata_directory="..\\..\\data\\Record Node 105\\experiment1\\recording1\\*.oebin",
        decompress_directory="..\\..\\data\\bombcell_decompress",
        bombcell_save_directory="..\\..\\data\\bombcell_quality_metrics",
        save_filename="bombcell_metrics.json"
    )
    tw = 2
