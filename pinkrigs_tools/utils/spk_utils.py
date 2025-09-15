import numpy as np
import pandas as pd

from floras_helpers.io import Bunch
from floras_helpers.hist import regions


def bombcell_sort_units_old(clusdat, max_peaks=2, max_throughs=1,
                        is_somatic=1, min_spatial_decay_slope=-0.003,
                        min_waveform_duration=100, max_waveform_duration=800,
                        max_waveform_baseline_fraction=.3, max_percentage_spikes_missing=20,
                        min_spike_num=300, max_refractory_period_violations=.1, min_amp=20, minSNR=.1, max_drift=500,
                        min_presence_ratio=.2):
    """
    classifier that sorts units into good,mua and noise, based on bombcell parameters

    Parameters: 
    ----------
    see bombcell documentation 


    Returns:
    -------
        	: np. array 
        bombcell class (mua/good/noise)
    """

    # any unit that is not discarded as noise or selected as well isolated is mua.
    # maybe there ought to be an option to classify well isolated axonal units...

    bombcell_class = np.empty(clusdat.nPeaks.size, dtype="object")
    bombcell_class[:] = 'mua'
    # assign noise 

    ix = (
            (clusdat.nPeaks > max_peaks) |
            (clusdat.nTroughs > max_throughs) |
            (clusdat.isSomatic != is_somatic) |
            (clusdat.spatialDecaySlope > min_spatial_decay_slope) |
            (clusdat.waveformDuration_peakTrough < min_waveform_duration) |
            (clusdat.waveformDuration_peakTrough > max_waveform_duration) |
            (clusdat.waveformBaselineFlatness > max_waveform_baseline_fraction)
    )

    bombcell_class[ix] = 'noise'

    # assign well isolated units 
    ix = (
            (bombcell_class != 'noise') &
            (clusdat.nSpikes >= min_spike_num) &
            (clusdat.fractionRPVs_estimatedTauR <= max_refractory_period_violations) &
            (clusdat.rawAmplitude >= min_amp) &
            (clusdat.percentageSpikesMissing_gaussian <= max_percentage_spikes_missing) &
            (clusdat.signalToNoiseRatio >= minSNR) &
            (clusdat.presenceRatio >= min_presence_ratio)
    )

    bombcell_class[ix] = 'good'

    return bombcell_class

# the new bombcell classifier -- fo now they need to co-exist because all data is in the old format except a few tested ones
# I moved this from the pyBombecell and rewrote it so that it takes the relevant params as kwargs similar to my old function
    
def bombcell_sort_units_new(clusdat,max_peaks=2, max_throughs=1, # waveform shapes
                        min_waveform_duration=100, max_waveform_duration=1150,max_waveform_baseline_fraction=.3, maxScndPeakToTroughRatio_noise=0.8, # waveform-based
                        computeSpatialDecay=True,min_spatial_decay_slope=0.01, max_spatial_decay_slope = 0.1, # spatial decay, based on exponential
                        minTroughToPeak2Ratio_nonSomatic=5, minWidthFirstPeak_nonSomatic=4, # non-somatic classification
                        minWidthMainTrough_nonSomatic=5, maxPeak1ToPeak2Ratio_nonSomatic=3, maxMainPeakToTroughRatio_nonSomatic=0.8, 
                        splitGoodAndMua_NonSomatic=True, # non-somatic classification
                        max_percentage_spikes_missing=20, min_spike_num=1500, max_refractory_period_violations = .5, # mua classification, changed rpv default from 0.1
                        min_presence_ratio=.7,
                        extractRaw=True, min_amp = 40, minSNR = 5, # raw waveform-based 
                        computeDrift=False, max_drift=100,
                        compute_distance_metrics=False, isodistance_min = 20, max_lratio = 0.3 # distance metrics    
    ):


        """
        Classifies neural units based on quality metrics.
        
        Unit Types:
        0: Noise units
        1: Good units
        2: MUA (Multi-Unit Activity)
        3: Non-somatic units (good if split)
        4: Non-somatic MUA (if split)

        Parameters
        ----------
        param : dict
            The parameters
        quality_metrics : dict
            The quality metrics
        
        Returns
        -------
        unit_type : ndarray
            The unit type classifaction as a number
        unit_type_string : ndarray
            The unit type classification as a string
        """
        n_units = len(clusdat["nPeaks"])
        unit_type = np.full(n_units, np.nan)
        
        # Noise classification
        noise_mask = (
            np.isnan(clusdat["nPeaks"]) |
            (clusdat["nPeaks"] > max_peaks) |
            (clusdat["nTroughs"] > max_throughs) |
            (clusdat["waveformDuration_peakTrough"] < min_waveform_duration) |
            (clusdat["waveformDuration_peakTrough"] > max_waveform_duration) |
            (clusdat["waveformBaselineFlatness"] > max_waveform_baseline_fraction) |
            (clusdat["scndPeakToTroughRatio"] > maxScndPeakToTroughRatio_noise)
        )

        if computeSpatialDecay:
            noise_mask |= (
                (clusdat["spatialDecaySlope"] < min_spatial_decay_slope) |
                (clusdat["spatialDecaySlope"] > max_spatial_decay_slope)
            )
        
        unit_type[noise_mask] = 0
        
        # Non-somatic classification
        is_non_somatic = (
            (clusdat["troughToPeak2Ratio"] < minTroughToPeak2Ratio_nonSomatic) &
            (clusdat["mainPeak_before_width"] < minWidthFirstPeak_nonSomatic) &
            (clusdat["mainTrough_width"] < minWidthMainTrough_nonSomatic) &
            (clusdat["peak1ToPeak2Ratio"] > maxPeak1ToPeak2Ratio_nonSomatic) |
            (clusdat["mainPeakToTroughRatio"] > maxMainPeakToTroughRatio_nonSomatic)
        )
        
        # MUA classification
        mua_mask = np.isnan(unit_type) & (
            (clusdat["percentageSpikesMissing_gaussian"] > max_percentage_spikes_missing) |
            (clusdat["nSpikes"] < min_spike_num) |
            (clusdat["fractionRPVs_estimatedTauR"] > max_refractory_period_violations) |
            (clusdat["presenceRatio"] < min_presence_ratio)
        )
        
        if extractRaw and np.all(~np.isnan(clusdat['rawAmplitude'])):
            mua_mask |= np.isnan(unit_type) & (
                (clusdat["rawAmplitude"] < min_amp) |
                (clusdat["signalToNoiseRatio"] < minSNR)
            )
        
        if computeDrift:
            mua_mask |= np.isnan(unit_type) & (clusdat["maxDriftEstimate"] > max_drift)
        
        if compute_distance_metrics:
            mua_mask |= np.isnan(unit_type) & (
                (clusdat["isolationDistance"] < isodistance_min) |
                (clusdat["Lratio"] > max_lratio)
            )
        
        unit_type[mua_mask] = 2
        unit_type[np.isnan(unit_type)] = 1
        
        # Handle non-somatic classifications
        if splitGoodAndMua_NonSomatic:
            good_non_somatic = (unit_type == 1) & is_non_somatic
            mua_non_somatic = (unit_type == 2) & is_non_somatic
            unit_type[good_non_somatic] = 3
            unit_type[mua_non_somatic] = 4
        else:
            unit_type[(unit_type != 0) & is_non_somatic] = 3
        
        # Create string labels
        labels = {0: "noise", 1: "good", 2: "mua", 
                3: "non-soma good" if splitGoodAndMua_NonSomatic else "non-soma",
                4: "non-soma mua"}
        
        unit_type_string = np.full(n_units, "", dtype=object)
        for code, label in labels.items():
            unit_type_string[unit_type == code] = label
        
        return unit_type, unit_type_string

def get_subregions(regionNames, mode='Beryl'):
    def classify_SC_acronym(allen_acronym):
        if ('SCs' in allen_acronym) or ('SCo' in allen_acronym) or ('SCzo' in allen_acronym):
            my_acronym = 'SCs'
        elif ('SCi' in allen_acronym):
            my_acronym = 'SCi'
        elif ('SCd' in allen_acronym):
            my_acronym = 'SCd'
        else:
            my_acronym = 'nontarget'
        return my_acronym

    if mode == "Beryl":
        reg = regions.BrainRegions()
        regionNames[regionNames == 'unregistered'] = 'void'
        parentregions = reg.acronym2acronym(regionNames, mapping='Beryl')

    elif mode == 'Cosmos':
        reg = regions.BrainRegions()
        regionNames[regionNames == 'unregistered'] = 'void'
        parentregions = reg.acronym2acronym(regionNames, mapping='Cosmos')

    elif '3SC' == mode:
        parentregions = np.array([classify_SC_acronym(n) for n in regionNames])

    return parentregions

def is_rec_in_region(rec, region_name='SC', framework='ccf', min_fraction=.1, goodOnly=False, **bombcell_kwargs):
    """
    utility function to assess whether a recording contains neurons in a target region
    
    Parameters:
    ----------
    rec: pd.Series
        typically the output of queryExp.load_data. Must contain probe.clusters

    region_name: str
        name of the region in AllenAcronym to match. Does not need to be at the exact level in the hierarchy as the Allen name is written out 
        (e.g. SC will count SCs & SCm)
    min_fraction: float
        min fraction of neurons that must be in the region so that the recording passes

    Returns:
    -------
        :bool
        whether rec is in region with region_name or not 
    """

    clusters = rec.probe.clusters
    print(rec.expFolder)
    if goodOnly:
        if 'isSomatic' in clusters.keys():            
            bc_class = bombcell_sort_units_old(clusters, **bombcell_kwargs)  # %%
        else:
            _ , bc_class = bombcell_sort_units_new(clusters, **bombcell_kwargs)
        is_good = bc_class == 'good'
        clusters = Bunch({k: clusters[k][is_good] for k in clusters.keys()})

    if min_fraction > 1:
        mode = 'number'
    else:
        mode = 'fraction'

    # check whether anatomy exists at all
    if 'mlapdv' not in list(clusters.keys()):
        is_region = False
    else:
        if framework == 'ccf':
            is_in_region = [region_name in x for x in clusters.brainLocationAcronyms_ccf_2017]
        else:
            region_names_in_framework = get_subregions(clusters.brainLocationAcronyms_ccf_2017, mode=framework)
            is_in_region = [x == region_name for x in region_names_in_framework]

        if (mode == 'fraction') & (np.mean(is_in_region) > min_fraction):
            is_region = True
        elif (mode == 'number') & (np.sum(is_in_region) > min_fraction):
            is_region = True

        else:
            is_region = False

    return is_region

def format_cluster_data(clusters):
    """
    this is a helper that further formats the cluster data, mostly with adding information about the anatomy and quality metrics
    
    """
    clusInfo = {k: clusters[k] for k in clusters.keys() if clusters[k].ndim == 1}
    clusInfo = pd.DataFrame.from_dict(clusInfo)
    # clusInfo = clusInfo.set_index('_av_IDs',drop=False)

    colnames = list(clusters.keys())
    if 'mlapdv' in colnames:
        # we could add the raw, but for now, I won't actually
        clusInfo['ml'] = clusters.mlapdv[:, 0]
        clusInfo['ap'] = clusters.mlapdv[:, 1]
        clusInfo['dv'] = clusters.mlapdv[:, 2]
        clusInfo['hemi'] = np.sign(clusInfo.ml - 5600)

    else:
        clusInfo['ml'] = np.nan
        clusInfo['ap'] = np.nan
        clusInfo['dv'] = np.nan
        clusInfo['hemi'] = np.nan
        clusInfo['brainLocationAcronyms_ccf_2017'] = 'unregistered'
        clusInfo['brainLocationIds_ccf_2017'] = np.nan

    if 'phy_clusterID' not in colnames:
        clusInfo['phy_clusterID'] = clusInfo.cluster_id

    br = regions.BrainRegions()
    if 'isSomatic' in clusInfo.keys():            
        bc_class = bombcell_sort_units_old(clusInfo)  # %%
    else:
        _ , bc_class = bombcell_sort_units_new(clusInfo)

    clusInfo['bombcell_class'] = bc_class
    clusInfo['is_good'] = bc_class == 'good'

    
    clusInfo.brainLocationAcronyms_ccf_2017[
        clusInfo.brainLocationAcronyms_ccf_2017 == 'unregistered'] = 'void'  # this is just so that the berylacronymconversion does something good
    clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')

    return clusInfo

def get_recorded_channel_position(channels):
    """
    todo: get IBL channels parameter. I think this needs to be implemented on the PinkRig level.
    """
    if not channels:
        xrange, yrange = None, None
    else:
        xcoords = channels.localCoordinates[:, 0]
        ycoords = channels.localCoordinates[:, 1]

        # if the probe is 3B, pyKS for some reason starts indexing from 1 depth higher (not 0)
        # to be fair that might be more fair, because the tip needs to be calculated to the anatomy 
        # alas who cares.
        if np.max(np.diff(ycoords)) == 20:
            # 3B probe
            ycoords = ycoords - ycoords[0]

        xrange = (np.min(xcoords), np.max(xcoords))
        yrange = (np.min(ycoords), np.max(ycoords))

    return (xrange, yrange)

