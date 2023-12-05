# Sourced from BombCell, https://github.com/Julie-Fabre/bombcell/blob/main/bc_qualityMetrics_pipeline.m
import json
import os
import sys
import warnings

MATLAB_SCRIPT = """
%% ~~ Example bombcell pipeline ~~
% Adjust the paths in the 'set paths' section and the parameters in bc_qualityParamValues
% This pipeline will:
%   (1) load your ephys data, 
%   (2) decompress your raw data if it is in .cbin format 
%   (3) run bombcell on your data and save the output and
%   (4) bring up summary plots and a GUI to flip through classified cells.
% The first time, this pipeline will be significantly slower (10-20' more)
% than after because it extracts raw waveforms. Subsequent times these
% pre-extracted waveforms are simply loaded in.
% We recommend running this pipeline on a few datasets and deciding on
% quality metric thresholds depending on the summary plots (histograms 
% of the distributions of quality metrics for each unit) and GUI. 

ephysKilosortPath = '{kilosort_directory}';% path to your kilosort output files 
ephysRawDir = dir('{raw_data_directory}'); % path to your raw .bin or .dat data
ephysMetaDir = dir('{metadata_directory}'); % path to your .meta or .oebin meta file
savePath = '{bombcell_save_directory}'; % where you want to save the quality metrics 
decompressDataLocal = '{decompress_directory}'; % where to save raw decompressed ephys data 

gain_to_uV = {gain_to_uv}; % use this if you are not using spikeGLX or openEphys to record your data. You then must leave the ephysMetaDir 
    % empty(e.g. ephysMetaDir = '')

%% check MATLAB version 
oldMATLAB = isMATLABReleaseOlderThan("R2019a");
if oldMATLAB
    error('This MATLAB version is older than 2019a - download a more recent version before continuing')
end

%% load data 
[spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, pcFeatures, ...
    pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

%% detect whether data is compressed, decompress locally if necessary
rawFile = bc_manageDataCompression(ephysRawDir, decompressDataLocal);

%% which quality metric parameters to extract and thresholds 
param = bc_qualityParamValues(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV); %for unitmatch, run this:
% param = bc_qualityParamValuesForUnitMatch(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV)


%% compute quality metrics 
rerun = 0;
qMetricsExist = ~isempty(dir(fullfile(savePath, 'qMetric*.mat'))) || ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));



if qMetricsExist == 0 || rerun
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
        templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions, savePath);
else
    [param, qMetric] = bc_loadSavedMetrics(savePath); 
    unitType = bc_getQualityUnitType(param, qMetric, savePath);
end


myjson = jsonencode(qMetric)
fp = fopen("{save_filename}", "wt");
fprintf(fp, myjson);


function bc_plotGlobalQualityMetric(varargin)
end
"""
if "matlabengine" not in sys.modules:
    warnings.warn("MatLabEngine python package not installed! Bombcell will not work!")

def bombcell_run_quality_metrics(kilosort_directory, raw_data_directory, metadata_directory, decompress_directory, bombcell_save_directory, save_filename, gain_to_uv=0.195):
    """
    Run quality metrics for bombcell, requires a MATLAB installation with the following add-on packages installed:
    - Signal Processing Toolbox
    - Image Processing Toolbox
    - Statistics and Machine Learning Toolbox
    - Parallel Computing Toolbox


    :param kilosort_directory: path to the output of kilosort. e.g. 'experiment1\\recording1\\continuous\\Neuropix-PXI-104.ProbeA-AP'
    :param raw_data_directory: path to the continuous.dat e.g. 'experiment1\\recording1\\continuous\\Neuropix-PXI-104-ProbeA-AP\\*.dat
    :param metadata_directory: path to .meta or .oebin e.g. 'experiment1\\recording1\\*.oebin'
    :param decompress_directory: path to a directory to use for decompressed waveforms
    :param bombcell_save_directory: path of a directory to save the quality metrics to, in raw .mat form
    :param gain_to_uv: Gain to uV
    :param save_filename: path of json file to save to
    :return:
    """
    if "matlabengine" not in sys.modules:
        raise ValueError("Cannot run bombcell without matlabengine python package installed!")

    matlab_code = MATLAB_SCRIPT.format(
        kilosort_directory=kilosort_directory,
        raw_data_directory=raw_data_directory,
        metadata_directory=metadata_directory,
        decompress_directory=decompress_directory,
        bombcell_save_directory=bombcell_save_directory,
        gain_to_uv=gain_to_uv,
        save_filename=save_filename
    )

    fp = open("bombcell_tmp_script.m", "w")
    fp.write(matlab_code)
    fp.close()

    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.bombcell_tmp_script(nargout=0)

    jfp = open(save_filename, "r")
    json_data = json.load(jfp)
    jfp.close()

    headers = [k for k in json_data[0].keys()]
    reformatted_json = {h: [] for h in headers}
    for entry in json_data:
        for h in headers:
            reformatted_json[h].append(entry[h])

    jfp = open(save_filename, "w")
    json.dump(reformatted_json, jfp)
    jfp.close()

    os.remove("bombcell_tmp_script.m")

    return json_data
