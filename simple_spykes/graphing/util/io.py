import warnings
from typing import Union, Optional
import numpy as np
import json


def load_file(metrics_file: Union[str, list[str]], exclude: Optional[list[str]] = None,
              use_common_units: bool = False) -> dict:
    """
    Load metric file(s) for use in graphing

    :param metrics_file: string filename to read from
    :param use_common_units: If true, will only use units that match up for each metric. (Sometimes certain metrics exclude units)
    :param exclude: List of string values from the qm file to exclude from loading
    :return: dict of QM formatted metric_data
    """
    if exclude is None:
        exclude = ["epoch_name", "cluster_id", "clusterID", "phy_clusterID"]

    if not isinstance(metrics_file, list):
        metrics_file = [metrics_file]

    data = {}
    for filename in metrics_file:
        with open(filename, "r") as f:
            loaddata = json.load(f)
            # loaddata = {"isolation_distance": loaddata["isolation_distance"]}
        [loaddata.pop(ex, None) for ex in exclude]
        data.update(loaddata)

    if not use_common_units:
        # Check that the metric_data of QMs have the same length, unless using common units then ignore
        val_len = None
        key_used_for_len = None
        for k, v in data.items():
            if val_len is None:
                val_len = len(v)
                key_used_for_len = k
            if len(v) != val_len:
                raise ValueError(f"Error, length of QM '{k}' isn't the same size as '{key_used_for_len}'")

    # Ensure common units align with each other
    all_set = None
    all_set_keyname = None
    for k, v in data.items():  # Keeping separate from above for loop so it can be easily commented out
        if all_set is None:
            all_set = set([int(s) for s in v.keys()])
            all_set_keyname = k
        cur_set = set([int(s) for s in v.keys()])
        if cur_set != all_set and not use_common_units:
            raise ValueError(
                f"Metrics do not have the same units! '{all_set_keyname}' and '{k}' To ignore set use_common_units=True")
        if use_common_units:
            all_set = all_set.intersection(cur_set)

    ordered_units = sorted(list(all_set))
    new_data = {}
    keylist = list(data.keys())

    for key in keylist:
        to_add = []
        for unit in ordered_units:
            to_add.append(data[key][str(unit)])
        to_add = np.array(to_add)
        if np.all(to_add == None) or np.all(to_add == 0) or np.all(to_add == to_add[0]):
            warnings.warn(f"All values of metric '{key}' are None, 0 or all the same! Excluding from graphing metric_data")
        else:
            new_data[key] = to_add
    return new_data
