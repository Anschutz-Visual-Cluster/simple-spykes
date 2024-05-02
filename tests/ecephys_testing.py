import os
from simple_spykes.metric_packages.ecephys_metrics import ecephys_run_quality_metrics


def ecephys_testing():
    os.chdir("metric_data")
    result = ecephys_run_quality_metrics(
        # Path to the output of kilosort, folder does NOT need to contain continuous.dat
        "E:\\QualityMetrics\\datasets\\josh\\curated\\2023-04-13",
        # float AP band sample rate in Hz, defaults to 30000
        30000.0,
        {
            "isi_threshold": 0.0015,
            "min_isi": 0.000166,
        },
        save_filename="ecephys_quality_metrics.json"
    )


# if __name__ == "__main__":
#     ecephys_testing()
