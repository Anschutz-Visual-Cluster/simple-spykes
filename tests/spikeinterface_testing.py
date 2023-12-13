from simple_spykes.metric_packages.spikeinterface_metrics import spikeinterface_run_quality_metrics


def spikeinterface_testing():
    # Data from \\10.33.107.246\s2\stim\recordings\jlh34_2023-05-15_16-03-21\Record Node 105\experiment1\recording1\continuous\Neuropix-PXI-104.ProbeA-AP
    r = spikeinterface_run_quality_metrics(
        folder_path="..\\..\\data\\Record Node 105",
        stream_name="Neuropix-PXI-104.ProbeA-AP",
        kilosort_output_directory="..\\..\\data\\Record Node 105\\experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP"
    )
