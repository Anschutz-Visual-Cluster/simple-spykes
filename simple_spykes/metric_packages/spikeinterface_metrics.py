import time
from pathlib import Path

import pendulum
from spikeinterface.extractors import read_kilosort
from spikeinterface.postprocessing import compute_principal_components, compute_spike_locations
from spikeinterface.qualitymetrics import compute_quality_metrics
import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter

from simple_spykes.util import save_json


def spikeinterface_run_quality_metrics(folder_path, stream_name, kilosort_output_directory, overwrite_waveform=False, save_filename=None):
    if save_filename is None:
        now = pendulum.now()
        save_filename = f"spikeinterface_quality_metrics-{now.month}-{now.day}-{now.year}_{now.hour}-{now.minute}-{now.second}.json"

    if not isinstance(save_filename, str) and (save_filename is not False):
        raise ValueError("Error, when specifying 'save_filename', value must be a string or False!")

    # Format the name of the stream as NEO expects it
    stream_name = f"{Path(folder_path).name}#{stream_name}"

    print("Reading open ephys")
    recording_extractor = si.read_openephys(folder_path, stream_name=stream_name)

    print("Reading kilosort")
    sorting_extractor = read_kilosort(kilosort_output_directory)
    recording_extractor = bandpass_filter(recording_extractor)

    print("Extracting waveforms")
    extracted_waveforms = si.extract_waveforms(
        recording_extractor,
        sorting_extractor,
        folder="QualityMetricsWaveforms",
        max_spikes_per_unit=None,  # Extract all spikes
        overwrite=overwrite_waveform,
        n_jobs=-1,  # Use all CPUs
        chunk_duration="1s",
        progress_bar=True,
        verbose=True,
        load_if_exists=not overwrite_waveform
    )

    extensions_to_load = {
        # https://github.com/SpikeInterface/spikeinterface/blob/3210f8eb960c404c91072596c39ef167af612353/src/spikeinterface/postprocessing/principal_component.py#L674
        "principal_components": [compute_principal_components, {
            "waveform_extractor": extracted_waveforms,  # Waveform extractor object above
            "n_components": 5,  # Number of components of PCA
            "n_jobs": -1,  # Number of jobs to fit, -1 uses all CPUs
            # mode: 'by_channel_local' a local PCA is fitted for each channel (projection by channel)
            # mode: 'by_channel_global' a global PCA is fitted for all channels (projection by channel)
            # mode: 'concatenated' channels are concatenated and a global PCA is fitted
            "mode": "by_channel_local",
            "load_if_exists": True,  # Load if exists
            "progress_bar": True,
            "verbose": True
        }],
        "spike_locations": [compute_spike_locations, {
            "waveform_extractor": extracted_waveforms,
            "load_if_exists": True,
            "ms_before": 0.5,  # The left window before a peak in ms
            "ms_after": 0.5,  # The right window after a peak in ms
            "method": "center_of_mass",  # "center_of_mass" | "monopolar_triangulation" | "grid_convolution"
            "progress_bar": True,
            "n_jobs": -1,
            "verbose": True
        }]
    }

    available_extensions = extracted_waveforms.get_available_extension_names()
    for extension_name, extension_data in extensions_to_load.items():
        if extension_name in available_extensions:
            print(f"Extension '{extension_name}' exists, loading..")
            extracted_waveforms.load_extension(extension_name)
        else:
            # Run the extension generating func with the args, see above dict
            print(f"Need to compute for extension '{extension_name}'..")
            extension_data[0](**extension_data[1])

    tw = 2

    # pca = compute_principal_components(waveform_extractor, n_components=5, mode='by_channel_local')
    """
    waveform_extractor,
    extractor obj
    
    load_if_exists=False,
        If True and pc scores are already in the waveform extractor folders, pc scores are loaded and not recomputed.
    
    n_components=5,
        Number of components fo PCA - default 5
    
    mode="by_channel_local",
        - 'by_channel_local': a local PCA is fitted for each channel (projection by channel)
        - 'by_channel_global': a global PCA is fitted for all channels (projection by channel)
        - 'concatenated': channels are concatenated and a global PCA is fitted
        
    sparsity=None,
        The sparsity to apply to waveforms, ChannelSparsity or None
        If waveform_extractor is already sparse, the default sparsity will be used - default None
        
    whiten=True,
         If True, waveforms are pre-whitened - default True
    
    dtype="float32",
        Dtype of the pc scores - default float32
    
    n_jobs=1,
        Number of jobs used to fit the PCA model (if mode is 'by_channel_local') - default 1
    
    progress_bar=False,
        If True, a progress bar is shown - default False
    
    tmp_folder=None,
        The temporary folder to use for parallel computation. If you run several `compute_principal_components`
        functions in parallel with mode 'by_channel_local', you need to specify a different `tmp_folder` for each call,
        to avoid overwriting to the same folder - default None
    
    """

    # metrics = compute_quality_metrics(waveform_extractor)
    """
    waveform_extractor,
    load_if_exists=False,
        Whether to load precomputed quality metrics, if they already exist.

    metric_names=None,
        List of quality metrics to compute.
            from spikeinterface.quality_metrics.quality_metric_list import _possible_pc_metric_names, _misc_metric_name_to_func
            --> use _misc_metric_name_to_func.keys()
            
            https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/qualitymetrics/pca_metrics.py#L35
            pc metrics
                "isolation_distance",
                "l_ratio",
                "d_prime",
                "nearest_neighbor",
                "nn_isolation",
                "nn_noise_overlap",
                "silhouette",         
            
            https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/qualitymetrics/quality_metric_list.py#L33
            misc_metrics dict
                "num_spikes": compute_num_spikes,
                "firing_rate": compute_firing_rates,
                "presence_ratio": compute_presence_ratios,
                "snr": compute_snrs,
                "isi_violation": compute_isi_violations,
                "rp_violation": compute_refrac_period_violations,
                "sliding_rp_violation": compute_sliding_rp_violations,
                "amplitude_cutoff": compute_amplitude_cutoffs,
                "amplitude_median": compute_amplitude_medians,
                "synchrony": compute_synchrony_metrics,
                "drift": compute_drift_metrics,
            
            
            PC REMOVED BY DEFAULT - MAYBE ADD?
                nn_noise_overlap
                nn_isolation
            
            
            
    qm_params=None,
        Dict of params for the given tests
            NON-PCA QM PARAMS
                _default_params["presence_ratio"] = dict(
                    bin_duration_s=60,
                    mean_fr_ratio_thresh=0.0,
                )
                _default_params["snr"] = dict(peak_sign="neg", peak_mode="extremum", random_chunk_kwargs_dict=None)
                _default_params["isi_violation"] = dict(isi_threshold_ms=1.5, min_isi_ms=0)
                _default_params["rp_violation"] = dict(refractory_period_ms=1.0, censored_period_ms=0.0)
                _default_params["sliding_rp_violation"] = dict(
                    min_spikes=0,
                    bin_size_ms=0.25,
                    window_size_s=1,
                    exclude_ref_period_below_ms=0.5,
                    max_ref_period_ms=10,
                    contamination_values=None,
                )
                _default_params["synchrony_metrics"] = dict(synchrony_sizes=(0, 2, 4))
                _default_params["amplitude_cutoff"] = dict(
                    peak_sign="neg", num_histogram_bins=100, histogram_smoothing_value=3, amplitudes_bins_min_ratio=5
                )
                _default_params["amplitude_median"] = dict(peak_sign="neg")
                _default_params["drift"] = dict(interval_s=60, min_spikes_per_interval=100, direction="y", min_num_bins=2)

            PCA QM Params
            dict(
                nearest_neighbor=dict(
                    max_spikes=10000,
                    n_neighbors=5,
                ),
                nn_isolation=dict(
                    max_spikes=10000, min_spikes=10, min_fr=0.0, n_neighbors=4, n_components=10, radius_um=100, peak_sign="neg"
                ),
                nn_noise_overlap=dict(
                    max_spikes=10000, min_spikes=10, min_fr=0.0, n_neighbors=4, n_components=10, radius_um=100, peak_sign="neg"
                ),
                silhouette=dict(method=("simplified",)),
            )

        
    sparsity=None,
        If given, the sparse channel_ids for each unit in PCA metrics computation.
        This is used also to identify neighbor units and speed up computations.
        If None (default) all channels and all units are used for each unit.
    verbose=False,  SET TO TRUE, shows more info    
        If True, output is verbose.
    progress_bar= SET TO True
        If True, progress bar is shown.
    """

    pc_metrics = [
        # PC Metrics

        "l_ratio",
        "d_prime",
        "nearest_neighbor",
        "nn_isolation",
        "nn_noise_overlap",
        "silhouette",
        "isolation_distance",
    ]

    non_pc_metrics = [
        # Non-PC Metrics
        "num_spikes",
        "firing_rate",
        "presence_ratio",
        "snr",
        "isi_violation",
        "rp_violation",
        "sliding_rp_violation",
        "amplitude_cutoff",
        "amplitude_median",
        "drift"
    ]
    all_metrics = [
        *non_pc_metrics,
        *pc_metrics
    ]

    all_metric_params = {
        # Non-PC Params
        "presence_ratio": {
            "bin_duration_s": 60,
            "mean_fr_ratio_thresh": 0.0
        },
        "snr": {
            "peak_sign": "neg",
            "peak_mode": "extremum",
            "random_chunk_kwargs_dict": None
        },
        "isi_violation": {
            "isi_threshold_ms": 1.5,
            "min_isi_ms": 0
        },
        "rp_violation": {
            "refractory_period_ms": 1.0,
            "censored_period_ms": 0.0
        },
        "sliding_rp_violation": {
            "min_spikes": 0,
            "bin_size_ms": 0.25,
            "window_size_s": 1,
            "exclude_ref_period_below_ms": 0.5,
            "max_ref_period_ms": 10,
            "contamination_values": None
        },
        "amplitude_cutoff": {
            "peak_sign": "neg",
            "num_histogram_bins": 100,
            "histogram_smoothing_value": 3,
            "amplitudes_bins_min_ratio": 5
        },
        "amplitude_median": {
            "peak_sign": "neg"
        },
        "drift": {
            "interval_s": 60,
            "min_spikes_per_interval": 100,
            "direction": "y",
            "min_num_bins": 2
        },

        # PC QM Params
        "nearest_neighbor": {
            "max_spikes": 10000,
            "n_neighbors": 5,
        },
        # NOTE this metric will take a long time
        "nn_isolation": {
            "max_spikes": 10000,
            "min_spikes": 10,
            "min_fr": 0.0,
            "n_neighbors": 4,
            "n_components": 10,
            "radius_um": 100,
            "peak_sign": "neg"
        },
        # NOTE this metric will take a long time
        "nn_noise_overlap": {
            "max_spikes": 10000,
            "min_spikes": 10,
            "min_fr": 0.0,
            "n_neighbors": 4,
            "n_components": 10,
            "radius_um": 100,
            "peak_sign":"neg"
        },
        "silhouette": {
            "method": ("simplified",)
        }
    }

    vals = compute_quality_metrics(
        extracted_waveforms,
        load_if_exists=False,
        metric_names=all_metrics,
        qm_params=all_metric_params,
        n_jobs=-1,  # use all CPUs
        skip_pc_metrics=False,
        progress_bar=True,
        verbose=True
    )

    json_data = vals.to_json()

    if save_filename:
        save_json(json_data, save_filename)

    return json_data

    # TODO Compute "synchrony" metrics?
    # https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/synchrony.html
    # "synchrony_metrics": {
    #     "synchrony_sizes": (0, 2, 4)
    # }
