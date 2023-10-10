import time
from pathlib import Path

from spikeinterface.extractors import read_kilosort
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.extractors import read_openephys
import spikeinterface.full as si
from spikeinterface.preprocessing import bandpass_filter


def run_quality_metrics(folder_path, stream_name, kilosort_output_directory):
    # Format the name of the stream as NEO expects it
    stream_name = f"{Path(folder_path).name}#{stream_name}"

    print("Reading open ephys")
    recording_extractor = si.read_openephys(folder_path, stream_name=stream_name)
    # probe = recording_extractor.get_probe()  # Automagically getting probe currently
    tw = 2
    # TODO add probe location and map, see slack from Juan
    print("Reading kilosort")
    sorting_extractor = read_kilosort(kilosort_output_directory)
    recording_extractor = bandpass_filter(recording_extractor)

    print("Extracting waveforms")
    extracted_waveforms = si.extract_waveforms(
        recording_extractor,
        sorting_extractor,
        folder="TestWaveform",
        max_spikes_per_unit=500,
        # overwrite=True,  # TODO Set this
        n_jobs=-1,  # Use all CPUs
        chunk_duration="1s",
        load_if_exists=True  # TODO Set this
    )

    available_extensions = extracted_waveforms.get_available_extension_names()
    if "principal_components" in available_extensions:
        print("PC extension exists, loading")
        extracted_waveforms.load_extension("principal_components")
    else:
        print("Computing PCs, REALLY SLOW! 72+ hours possibly!")
        compute_principal_components(waveform_extractor=extracted_waveforms, n_components=5,
                                           mode='by_channel_local')

    tw = 2

    # https://github.com/SpikeInterface/spikeinterface/blob/3210f8eb960c404c91072596c39ef167af612353/src/spikeinterface/postprocessing/principal_component.py#L674
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

    # https://github.com/SpikeInterface/spikeinterface/blob/3210f8eb960c404c91072596c39ef167af612353/src/spikeinterface/qualitymetrics/quality_metric_calculator.py#L176
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

    num_units = len(extracted_waveforms.unit_ids)
    units_per_iteration = 20
    groups = num_units // units_per_iteration
    extra = num_units % units_per_iteration
    group_sizes = list(range(0, num_units, units_per_iteration))
    if extra != 0:
        group_sizes.append(num_units)
    # unit_groups = [list(range(group_sizes[i-1], group_sizes[i])) for i in range(1, len(group_sizes))]
    unit_groups = [list(range(0, num_units))]
    tw = 2

    print("Computing different quality metrics")
    for metric in pc_metrics:
        print(f"Working on metric '{metric}'")
        start = time.time()
        for unit_group in unit_groups:
            print(f"Working on unit group {unit_group[0]}-{unit_group[-1]}")
            vals = compute_quality_metrics(
                extracted_waveforms,
                metric_names=[metric],  # TODO change metric
                qm_params=all_metric_params,
                # unit_ids=unit_group,  # TODO Remove, keyword doesn't exist
                n_jobs=-1  # use all CPUs
            )
        print(f"Finished '{metric}' in {time.time() - start}")

    # TODO Compute "synchrony" metrics
    # https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/synchrony.html
    # "synchrony_metrics": {
    #     "synchrony_sizes": (0, 2, 4)
    # },
    tw = 2

    pass
