import json
import time
import uuid
import warnings

import numpy as np
import pendulum
from ecephys_spike_sorting.common.utils import load_kilosort_data
from ecephys_spike_sorting.modules.quality_metrics.metrics import calculate_metrics

from simple_spykes.util import save_json

"""
Code Sourced from
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/scripts/batch_processing.py
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/__main__.py

Adapted to fit needs
"""


def ecephys_run_quality_metrics(kilosort_output_directory, sample_rate, quality_metrics_params, save_filename=None):
    """
    Run EcePhys quality metrics
    Input for Metrics params looks like (example values below)
    {
            "isi_threshold": 0.0015,
            "min_isi": 0.000166
    }

    Output looks like:
    {
        "cluster_id": { id str: int, ...},
        "firing_rate": { id str: float, ...},
        "isi_viol": { id str: float, ...},
        "amplitude_cutoff": { id str: float, ...},
        "epoch_name": { id str: str, ...}
    }

    :param kilosort_output_directory:
        Path to the output of kilosort, folder containing continuous.dat
        eg "E:\\NeuroPixelsTest\\continuous\\Neuropix-PXI-104.ProbeA-AP"

    :param sample_rate: AP band sample rate in Hz
    :param quality_metrics_params: Parameters for the quality metrics tests, see above for details
    :param save_filename: string to save to file, Set to False to ignore
    :return: json dict of the quality metrics values
    """
    warnings.warn("The EcePhys package isn't well optimized, consider using another package.")
    if save_filename is None:
        now = pendulum.now()
        save_filename = f"ecephys_quality_metrics-{now.month}-{now.day}-{now.year}_{now.hour}-{now.minute}-{now.second}.json"

    if not isinstance(save_filename, str) and (save_filename is not False):
        raise ValueError("Error, when specifying 'save_filename', value must be a string or False!")

    start = time.time()
    print("Loading metric_data...")
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

        print('Total time: ' + str(np.around(time.time() - start, 2)) + ' seconds')
        json_data = metrics.to_json()

        if save_filename:
            save_json(json_data, save_filename)

        return json.loads(json_data)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot find file needed to run quality metrics! Error: {str(e)}")
