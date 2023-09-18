# from simple_spykes.util.ecephys import run_quality_metrics
from simple_spykes.util.spikeinterface import run_quality_metrics

KILOSORT_OUTPUT_DIR = "E:\\Record Node 103\\experiment1\\recording1\\continuous\\Neuropix-PXI-102.ProbeA-AP"


def main():
    r = run_quality_metrics(
        folder_path="E:\\Record Node 103",
        stream_name = "Neuropix-PXI-102.ProbeA-AP",
        kilosort_output_directory=KILOSORT_OUTPUT_DIR)

    tw = 2


if __name__ == "__main__":
    main()
