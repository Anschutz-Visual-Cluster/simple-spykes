import tempfile

from ecephys_spike_sorting.scripts.create_input_json import createInputJson
import os
import subprocess
"""
Code Sourced from https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/scripts/batch_processing.py
Adapted to fit needs
"""


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
    config = generate_input_json()
    tw = 2


if __name__ == "__main__":
    main()
