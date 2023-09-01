"""
Code Sourced from
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/scripts/batch_processing.py
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/__main__.py

Adapted to fit needs
"""
from simple_spykes.util.ecephys import run_quality_metrics


def generate_input_json():
    # createInputJson(output_file,
    #                 npx_directory=None,
    #                 continuous_file=None,
    #                 extracted_data_directory=None,
    #                 kilosort_output_directory=None,
    #                 kilosort_output_tmp=None,
    #                 probe_type='3A')
    from ecephys_spike_sorting.scripts.create_input_json import createInputJson

    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    json_data = createInputJson(tmpfile.name, extracted_data_directory="E:\\NeuroPixelsTest")
    tmpfile.close()
    os.remove(tmpfile.name)

    return json_data
    pass


def main():
    # config = generate_input_json()
    result = run_quality_metrics(
        # Path to the output of kilosort, folder containing continuous.dat
        "E:\\NeuroPixelsTest\\continuous\\Neuropix-PXI-104.ProbeA-AP",
        # float (optional) AP band sample rate in Hz, defaults to 30000
        30000.0,
        {
            "isi_threshold": 0.0015,
            "min_isi": 0.000166,
            "num_channels_to_compare": 7,
            "max_spikes_for_unit": 500,
            "max_spikes_for_nn": 10000,
            "n_neighbors": 4,
            'n_silhouette': 10000,
            "drift_metrics_interval_s": 51,
            "drift_metrics_min_spikes_per_interval": 10
        },
        save_to_file="quality_metrics.json"
    )

    tw = 2


if __name__ == "__main__":
    main()
