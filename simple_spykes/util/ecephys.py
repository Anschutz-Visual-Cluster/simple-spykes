import json
import time
import uuid
import numpy as np
from ecephys_spike_sorting.common.utils import load_kilosort_data
from ecephys_spike_sorting.modules.quality_metrics.metrics import calculate_metrics


def run_quality_metrics(kilosort_output_directory, sample_rate, quality_metrics_params, save_to_file=None):
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
            include_pcs=False
        )
        print("Unpacking and starting calculations..")

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
