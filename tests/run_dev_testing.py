import tempfile
import time
import numpy as np
import os
from ecephys_spike_sorting.scripts.create_input_json import createInputJson
from ecephys_spike_sorting.common.utils import load_kilosort_data
from ecephys_spike_sorting.modules.quality_metrics.metrics import calculate_metrics

"""
Code Sourced from
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/scripts/batch_processing.py
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/__main__.py

Adapted to fit needs
"""


def run_quality_metrics(kilosort_output_directory, sample_rate, quality_metrics_params):
    print('ecephys spike sorting: quality metrics module')
    start = time.time()

    print("Loading data...")
    try:
        spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality, pc_features, pc_feature_ind = \
            load_kilosort_data(
                kilosort_output_directory,
                sample_rate,
                use_master_clock=False,
                include_pcs=True
            )

        pc_features = None
        pc_feature_ind = None

        metrics = calculate_metrics(spike_times,
                                    spike_clusters,
                                    spike_templates,
                                    amplitudes,
                                    channel_map,
                                    pc_features,
                                    pc_feature_ind,
                                    quality_metrics_params)
    except FileNotFoundError:
        execution_time = time.time() - start
        print(" Files not available.")
        return {"execution_time": execution_time,
                "quality_metrics_output_file": None}
    #
    # output_file = args['quality_metrics_params']['quality_metrics_output_file']
    #
    # if os.path.exists(args['waveform_metrics']['waveform_metrics_file']):
    #     metrics = metrics.merge(pd.read_csv(args['waveform_metrics']['waveform_metrics_file'], index_col=0),
    #                             on='cluster_id',
    #                             suffixes=('_quality_metrics', '_waveform_metrics'))
    #
    # print("Saving data...")
    # tw = 2
    # # metrics.to_csv(output_file)

    execution_time = time.time() - start

    print('total time: ' + str(np.around(execution_time, 2)) + ' seconds')
    print()

    return {"execution_time": execution_time,
            "quality_metrics_output_file": "adlfkjasasdf"}  # output manifest


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
        }
    )

    tw = 2


if __name__ == "__main__":
    main()
