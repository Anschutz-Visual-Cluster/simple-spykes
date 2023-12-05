from simple_spykes.metric_packages import bombcell_run_quality_metrics
import os
import shutil


PREFIX = "D:\\QualityMetrics\\Datasets\\josh"

CURATED_DIRECTORY = os.path.join(PREFIX, "curated")
RAW_DIRECTORY = os.path.join(PREFIX, "raw")
USE_DATE_INTERSECTION = True
TEMP_DIRECTORY = "D:\\tmp"


def clean_tmp_dir():
    print(f"Cleaning up temp dir '{TEMP_DIRECTORY}'..")
    shutil.rmtree(TEMP_DIRECTORY)
    os.mkdir(TEMP_DIRECTORY)
    print("Done cleaning up temp dir")


def main():
    print("Sleeping for an hour to wait for the copy to finish")
    import time
    time.sleep(60*60)

    print("Testing raw")
    test_raw_dir = os.path.join(RAW_DIRECTORY, "2023-04-11", "continuous", "Neuropix-PXI-100.ProbeA-AP")
    bombcell_run_quality_metrics(
        kilosort_directory=test_raw_dir,
        raw_data_directory=os.path.join(test_raw_dir, "continuous.dat"),
        metadata_directory=os.path.join(RAW_DIRECTORY, "2023-04-11", "structure.oebin"),
        decompress_directory=TEMP_DIRECTORY,
        bombcell_save_directory=TEMP_DIRECTORY,
        save_filename="2023-04-11_testing_raw_metrics.json"
    )
    clean_tmp_dir()

    print("Testing curated")
    test_curated_dir = os.path.join(RAW_DIRECTORY, "2023-04-11")
    bombcell_run_quality_metrics(
        kilosort_directory=test_curated_dir,
        raw_data_directory=os.path.join(test_curated_dir, "continuous.dat"),
        metadata_directory=os.path.join(test_curated_dir, "structure.oebin"),
        decompress_directory=TEMP_DIRECTORY,
        bombcell_save_directory=TEMP_DIRECTORY,
        save_filename="2023-04-11_testing_curated_metrics.json"
    )
    clean_tmp_dir()

    list_of_dates = os.listdir(CURATED_DIRECTORY)
    raw_dates = os.listdir(RAW_DIRECTORY)

    if list_of_dates != raw_dates:
        curated_only = set(list_of_dates).difference(set(raw_dates))
        raw_only = set(raw_dates).difference(set(list_of_dates))
        both = set(list_of_dates).intersection(set(raw_dates))
        if USE_DATE_INTERSECTION:
            list_of_dates = list(both)
        else:
            raise ValueError(f"Curated vs Raw don't have the same values! Curated only: '{curated_only}' Raw only: " +
                             f"'{raw_only}' Both: '{both}' To ignore this and use both, set USE_DATE_INTERSECTION" +
                             "to True")
    print(f"Running quality metrics on date folders: '{list_of_dates}'")

    for date in list_of_dates:
        print(f"Processing '{date}'")
        # grab something like 'Neuropix-PXI-104.ProbeA-AP'
        probe_folder_name = os.listdir(os.path.join(RAW_DIRECTORY, date, "continuous"))[0]

        params = {
            "kilosort_directory": probe_folder_name,
            "raw_data_directory": os.path.join(probe_folder_name, "continuous.dat"),  # TODO change up dir structure of raw / autogenerate correct dirs?
            "metadata_directory": os.path.join(RAW_DIRECTORY, "structure.oebin"),
            "decompress_directory": TEMP_DIRECTORY,
            "bombcell_save_directory": TEMP_DIRECTORY,
            "save_filename": f"bombcell-{date}-raw.json"
        }

        print("Processing RAW")
        bombcell_run_quality_metrics(**params)
        clean_tmp_dir()

        print("Processing CURATED")
        params["kilosort_directory"] = os.path.join(RAW_DIRECTORY, date)  # TODO change up dir structure of curated?
        params["save_filename"] = f"bombcell-{date}-curated.json"

        bombcell_run_quality_metrics(**params)
        clean_tmp_dir()
    print("Done!")


if __name__ == "__main__":
    main()
