import json
import tempfile
import time
import uuid

import numpy as np
import os

# Suppress Git error when importing from ecephys, so it doesn't throw an exception
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from ecephys_spike_sorting.scripts.create_input_json import createInputJson
from ecephys_spike_sorting.common.utils import load_kilosort_data
from ecephys_spike_sorting.modules.quality_metrics.metrics import calculate_metrics

"""
Code Sourced from
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/scripts/batch_processing.py
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/__main__.py

Adapted to fit needs
"""


def run_quality_metrics(kilosort_output_directory, sample_rate, quality_metrics_params, save_to_file=None, calc_pcs=True):
    if save_to_file is None and not isinstance(save_to_file, str):
        raise ValueError("Error, when specifying 'save_to_file', value must be a string!")

    print('ecephys spike sorting: quality metrics module')
    start = time.time()

    print("Loading data...")
    try:
        load_result = load_kilosort_data(
            kilosort_output_directory,
            sample_rate,
            use_master_clock=False,
            include_pcs=calc_pcs
        )
        print("Unpacking and starting calculations..")

        if calc_pcs:
            spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality, pc_features, pc_feature_ind = load_result
        else:
            pc_features = None
            pc_feature_ind = None
            spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality = load_result

        metrics = calculate_metrics(spike_times,
                                    spike_clusters,
                                    spike_templates,
                                    amplitudes,
                                    channel_map,
                                    pc_features,
                                    pc_feature_ind,
                                    quality_metrics_params)

        print('total time: ' + str(np.around(time.time() - start, 2)) + ' seconds')
        json_data = metrics.to_json()
        if save_to_file:
            other_filename = f"quality_metrics_{str(uuid.uuid4())}"
            try:
                print(f"Saving metrics to file '{save_to_file}'")
                fp = open(save_to_file, "w")
                fp.write(json_data)
                fp.close()
            except Exception as e:
                print(f"Error saving metrics to specified file '{save_to_file}'! Saving to file '{other_filename}' \nError: {str(e)}")
                fp = open(other_filename, "w")
                fp.write(json_data)
                fp.close()

        return json.loads(json_data)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot find file needed to run quality metrics! Error: {str(e)}")


def generate_input_json():
    # createInputJson(output_file,
    #                 npx_directory=None,
    #                 continuous_file=None,
    #                 extracted_data_directory=None,
    #                 kilosort_output_directory=None,
    #                 kilosort_output_tmp=None,
    #                 probe_type='3A')

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
            "drift_metrics_min_spikes_per_interval": 10,
            "include_pc_metrics": True
        },
        save_to_file="quality_metrics.json",
        calc_pcs=False
    )

    tw = 2


if __name__ == "__main__":
    main()
