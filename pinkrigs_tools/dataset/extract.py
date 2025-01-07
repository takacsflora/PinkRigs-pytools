# this code queries and extracts av data, 
# more specifically will output a folder of csvs where rows are trials and columns are 
# ev information as well as extracted neural activity
# maybe this collection should also contain the design matrix formatter..? up to decision for later

# 

#%%

# this 
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from pinkrigs_tools.dataset.pre_cured import call_
from pinkrigs_tools.utils.ev_utils import get_triggered_data_per_trial
from pinkrigs_tools.utils.spk_utils import format_cluster_data

# from loaders import load_params
#from predChoice import format_av_trials
# this queries the csv for possible recordings 

my_ROI = 'SCm'
paramset_name = 'choice'

savepath = Path(r'D:\AVTrialData\%s_%s' % (my_ROI,paramset_name))


pre_cured_call_args = {
    'subject_set': 'AV034',
    'spikeToInclde': True,
    'camToInclude': True,
    'camPCsToInclude': False,
    'recompute_data_selection': True,
    'unwrap_probes': False,
    'merge_probes': True,
    'filter_unique_shank_positions': False,
    'region_selection': {
        'region_name': my_ROI,
        'framework': 'Beryl',
        'min_fraction': 20,
        'goodOnly': True,
        'min_spike_num': 300
    },
    'min_rt': 0,
    'analysis_folder': savepath
}

trigger_timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0,
            'post_time':0.15
        }

active_sessions  = call_(dataset_type = 'active', **pre_cured_call_args)
passive_sessions  = call_(dataset_type = 'postactive', **pre_cured_call_args)

# 
def generate_meta_data(rec):
    pass

#%%


trial_path = savepath / 'trial_data'
cluster_path  = savepath / 'cluster_data'

trial_path = Path(trial_path)
trial_path.mkdir(parents=False,exist_ok=True)

cluster_path = Path(cluster_path)
cluster_path.mkdir(parents=False,exist_ok=True)




for _,rec in active_sessions.iterrows():
    sessname = '{subject}_{expDate}.csv'.format(**rec) # will be the matching data file
    # corresponding passive session
    passive_rec = passive_sessions.query('(subject == @rec.subject) & (expDate == @rec.expDate)').iloc[0]

    # save cluster data
    
    cluster_data_active = format_cluster_data(rec.probe.clusters)
    cluster_data_passive = format_cluster_data(passive_rec.probe.clusters)

    assert cluster_data_active.shape[0] == cluster_data_passive.shape[0],'something iffy'

    cluster_data = cluster_data_active.copy() 
    cluster_data.to_csv((cluster_path / sessname),index=False)

    #save spike data
    trial_data_active = get_triggered_data_per_trial(
        ev= rec.events._av_trials,
        spikes=rec.probe.spikes,
        nID=cluster_data._av_IDs.values,
        cam=rec.camera,
        **trigger_timing_params
    ) 


    trial_data_passive = get_triggered_data_per_trial(
        ev= passive_rec.events._av_trials,
        spikes=rec.probe.spikes,
        nID=cluster_data._av_IDs.values,
        cam=rec.camera,
        **trigger_timing_params
    ) 

    trial_data = pd.concat([trial_data_active,trial_data_passive],axis=0)
    trial_data.to_csv((trial_path / sessname),index=False)
    

    # save_out_meta_data



#%%
# first I sort out the way I want to extract the data, thne I will add the passive sessions
#passive_sessions  = call_(dataset_type = 'postactive', **pre_cured_call_args)



#  not sure what this is for

# neuronNos =[]
# for _,rec in selected_sessions.iterrows():
#     ev,spk,clusInfo,_,cam = simplify_recdat(rec,probe='probe')
#     goodclusIDs = clusInfo[(clusInfo.is_good)&(clusInfo.BerylAcronym==my_ROI)]._av_IDs.values
#     print(
#         rec.expFolder,
#         ev.is_auditoryTrial.size,
#         goodclusIDs.size, 
#     )
#     neuronNos.append(goodclusIDs.size)
    
# selected_sessions['neuronNo'] = neuronNos

# # %
# # Group by 'subject' and aggregate
# result = selected_sessions.groupby('subject').agg(
#     session_count=('subject', 'size'),  # Count the number of rows per subject
#     neuron_sum=('neuronNo', 'sum')  # Sum the 'neuronNo' values per subject
# ).reset_index()

# print(result)
