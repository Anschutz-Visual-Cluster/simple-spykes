from simple_spykes.metric_packages.ecephys_metrics import ecephys_run_quality_metrics


def test_ecephys():
    result = ecephys_run_quality_metrics(
        # Path to the output of kilosort, folder containing continuous.dat
        "E:\\NeuroPixelsTest\\continuous\\Neuropix-PXI-104.ProbeA-AP",
        # float (optional) AP band sample rate in Hz, defaults to 30000
        30000.0,
        {
            "isi_threshold": 0.0015,
            "min_isi": 0.000166,

            # PC Metrics are disabled as they consistently fail to run / aren't efficient
            # # Pc metrics isolation distance, l_ratio, d_primt, nn_hit_rate, nn_mis_rate
            # "num_channels_to_compare": 7,
            # "max_spikes_for_unit": 500,
            # "max_spikes_for_nn": 10000,
            # "n_neighbors": 4,

            # # Silhouette score
            # 'n_silhouette': 10000,
            # # Drift metrics (max drift, cumulative drift)
            # "drift_metrics_interval_s": 51,
            # "drift_metrics_min_spikes_per_interval": 10

            # Including Doc example in here in case we ever re-enable PC metrics
            # "num_channels_to_compare": 7,
            # "max_spikes_for_unit": 500,
            # "max_spikes_for_nn": 10000,
            # "n_neighbors": 4,
            # 'n_silhouette': 10000,
            # "drift_metrics_interval_s": 51,
            # "drift_metrics_min_spikes_per_interval": 10

        },
        save_to_file="quality_metrics.json"
    )
