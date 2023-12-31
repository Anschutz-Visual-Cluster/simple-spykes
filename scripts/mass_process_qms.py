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
    shutil.rmtree(TEMP_DIRECTORY, ignore_errors=True)
    os.mkdir(TEMP_DIRECTORY)
    print("Done cleaning up temp dir")


def run_bombcell(params, date_name, suffix):
    try:
        bombcell_run_quality_metrics(**params)
    except Exception as e:
        print(f"Failed on {date_name} with error {str(e)}")
        with open(f"error-{date_name}-{suffix}.txt", "w") as f:
            f.write(str(e))
    clean_tmp_dir()


def main():
    # print("Testing raw")
    # test_date = "2023-07-24"
    # test_raw_dir = os.path.join(RAW_DIRECTORY, test_date, "continuous", "Neuropix-PXI-100.ProbeA-AP")
    # bombcell_run_quality_metrics(
    #     kilosort_directory=test_raw_dir,
    #     raw_data_directory=os.path.join(test_raw_dir, "continuous.dat"),
    #     metadata_directory=os.path.join(RAW_DIRECTORY, test_date, "structure.oebin"),
    #     decompress_directory=TEMP_DIRECTORY,
    #     bombcell_save_directory=TEMP_DIRECTORY,
    #     save_filename=f"{test_date}_testing_raw_metrics.json"
    # )
    # clean_tmp_dir()
    #
    # print("Testing curated")
    # test_curated_dir = os.path.join(CURATED_DIRECTORY, test_date)
    # bombcell_run_quality_metrics(
    #     kilosort_directory=test_curated_dir,
    #     raw_data_directory=os.path.join(test_curated_dir, "continuous.dat"),
    #     metadata_directory=os.path.join(test_curated_dir, "structure.oebin"),
    #     decompress_directory=TEMP_DIRECTORY,
    #     bombcell_save_directory=TEMP_DIRECTORY,
    #     save_filename=f"{test_date}_testing_curated_metrics.json"
    # )
    # clean_tmp_dir()

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
        raw_base_folder = os.path.join(RAW_DIRECTORY, date, "continuous")
        probe_folder_name = os.path.join(raw_base_folder, os.listdir(raw_base_folder)[0])

        params = {
            "kilosort_directory": probe_folder_name,
            "raw_data_directory": os.path.join(probe_folder_name, "continuous.dat"),  # TODO change up dir structure of raw / autogenerate correct dirs?
            "metadata_directory": os.path.join(RAW_DIRECTORY, "structure.oebin"),
            "decompress_directory": TEMP_DIRECTORY,
            "bombcell_save_directory": TEMP_DIRECTORY,
            "save_filename": f"bombcell-{date}-raw.json"
        }

        # print("Processing RAW")
        # run_bombcell(params, date, "raw")
        # clean_tmp_dir()

        print("Processing CURATED")
        params["kilosort_directory"] = os.path.join(CURATED_DIRECTORY, date)  # TODO change up dir structure of curated?
        params["save_filename"] = f"bombcell-{date}-curated.json"

        run_bombcell(params, date, "curated")
        clean_tmp_dir()

    print("Done!")


if __name__ == "__main__":
    main()
