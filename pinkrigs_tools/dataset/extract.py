# this code queries and extracts av data, 
# more specifically will output a folder of csvs where rows are trials and columns are 
# ev information as well as extracted neural activity
# maybe this collection should also contain the design matrix formatter..? up to decision for later

#    


import numpy as np
import pandas as pd
from pathlib import Path

from pinkrigs_tools.dataset.pre_cured import call_
from pinkrigs_tools.utils.ev_utils import get_triggered_data_per_trial
from pinkrigs_tools.utils.spk_utils import format_cluster_data

from floras_helpers.io import Bunch,save_dict_to_json


def get_params(paramset='choice'):
    """_summary_

    Args:
        paramset (str, optional): _description_. Defaults to 'choice'.

    Returns:
        _type_: _description_
    """
    if paramset == 'choice':

        timing_params = {
            'onset_time':'timeline_choiceMoveOn',
            'pre_time':0.15,
            'post_time':0
        }

    elif paramset == 'prestim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0.15,
            'post_time':0
        }
    
    elif paramset == 'poststim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0,
            'post_time':0.15
        }
    
    return timing_params

def generate_meta_data(rec_list):
    """
    could be better aha

    """

    rec = rec_list[0] 
    passive_rec = rec_list[1]

    return({
            'subject': rec.subject,
            'expDate': rec.expDate,
            'rigName': rec.rigName,
            'passive_expFolder': passive_rec.expFolder,
            'active_expFolder': rec.expFolder
        })

def cumulate_timings(spikes_list,ev_list,cam_list):
    # get the max spike time for each session
    max_spike_time = np.array([spikes.times.max() for spikes in spikes_list]) 
    # pop in a 0 as the 1st entry
    max_spike_time = np.insert(max_spike_time,0,0)
    # cumulate the max spike time
    cumulative_spike_time = np.cumsum(max_spike_time) # could do this with rec.expDuration too... 
    # for each spike time in the list, add the cumulative spike time
    for i,spikes in enumerate(spikes_list):
        spikes.times += cumulative_spike_time[i]
        # add same for all ev times in ev_list, i.e. keys that have 'On' or 'Off' in their names
        for key in ev_list[i].keys():
            if 'On' in key or 'Off' in key:
                ev_list[i][key] += cumulative_spike_time[i]

        # add same for all cam times in cam_list
        cam_list[i].times += cumulative_spike_time[i]


    return spikes_list,ev_list,cam_list

def extract_cluster_data(clusters_list):
    cluster_data = [format_cluster_data(clusters) for clusters in clusters_list]
    
    assert all(df.shape[0] == cluster_data[0].shape[0] for df in cluster_data), 'not the same clusters are passed down to match, this is gonna be hard...'

    return cluster_data[0].copy()

def combine_spikes(spikes_list):
    spikes_combined = {}
    for key in spikes_list[0].keys():
        spikes_combined[key] = np.concatenate([spikes[key] for spikes in spikes_list])

    return pd.DataFrame.from_dict(spikes_combined) 

def init_folder_structure(savepath):

    trial_path = savepath / 'trial_data'
    cluster_path  = savepath / 'cluster_data'
    spikes_path = savepath / 'spikes_data'
    meta_path = savepath / 'meta_data'

    trial_path = Path(trial_path)
    trial_path.mkdir(parents=False,exist_ok=True)

    cluster_path = Path(cluster_path)
    cluster_path.mkdir(parents=False,exist_ok=True)

    spikes_path = Path(spikes_path)
    spikes_path.mkdir(parents=False,exist_ok=True)

    meta_path = Path(meta_path)
    meta_path.mkdir(parents=False,exist_ok=True)


    return trial_path,cluster_path,spikes_path,meta_path

def prproc_and_save(brain_region=None,
                    paramset_name='poststim',
                    subject_set = 'active',
                    recompute_data_selection = False):

    savepath = Path(r'D:\AVTrialData\%s_%s' % (brain_region,paramset_name))

    data_call_arguments = {
        'subject_set': subject_set,
        'spikeToInclde': True,
        'camToInclude': True,
        'camPCsToInclude': False,
        'recompute_data_selection': recompute_data_selection,
        'unwrap_probes': False,
        'merge_probes': True,
        'filter_unique_shank_positions': False,
        'region_selection': {
            'region_name': brain_region,
            'framework': 'Beryl',
            'min_fraction': 20,
            'goodOnly': True,
            'min_spike_num': 300
        },
        'min_rt': 0,
        'analysis_folder': savepath
    }

# nvm I need to update this one!

    trigger_timing_params = get_params(paramset=paramset_name)

    active_sessions  = call_(dataset_type = 'active', **data_call_arguments)
    passive_sessions  = call_(dataset_type = 'postactive', **data_call_arguments)

    meta_data = []
    trial_path,cluster_path,spikes_path,meta_path = init_folder_structure(savepath)
    for _,rec in active_sessions.iterrows():
        sessname = '{subject}_{expDate}.csv'.format(**rec) # will be the matching data file
        
        
        # find the corresponding passive session
        passive_rec_query = passive_sessions.query('(subject == @rec.subject) & (expDate == @rec.expDate)')
        
        if passive_rec_query.shape[0] == 0:
            print('no passive session found for %s' % sessname)
        
        else:
            passive_rec = passive_rec_query.iloc[0]



            # build a list of all the data we are extracting  (btw this is potentailly how ut would work if we concatenate several days)
            clusters_list = [rec.probe.clusters,passive_rec.probe.clusters]
            spikes_list = [rec.probe.spikes,passive_rec.probe.spikes]
            ev_list  = [rec.events._av_trials,passive_rec.events._av_trials]
            cam_list = [rec.camera,passive_rec.camera]

            spikes_list,ev_list,cam_list = cumulate_timings(spikes_list,ev_list,cam_list)


            cluster_data = extract_cluster_data(clusters_list)
            clusIDs = cluster_data._av_IDs.values
            

            # get the trigged data
            trial_data = pd.concat([get_triggered_data_per_trial(
                                ev=ev,
                                spikes= spk,
                                nID = clusIDs,cam=cam,
                                single_average_accross_neurons=False,
                                get_zscored=False,
                                **trigger_timing_params
                                ) 
                            
                            for ev,spk,cam in zip(ev_list,spikes_list,cam_list)],axis=1)

            spikes_data = combine_spikes(spikes_list)

            # Save the data
            spikes_data.to_csv((spikes_path / sessname),index=False)
            cluster_data.to_csv((cluster_path / sessname),index=False)
            trial_data.to_csv((trial_path / sessname),index=False)          

            meta_data.append(generate_meta_data([rec,passive_rec]))

    meta_df = pd.DataFrame([meta_data])

    meta_df.to_csv(meta_path / 'sessInfo.csv',index=False)
    save_dict_to_json(trigger_timing_params,meta_path / 'trigger_timing_params.json')
    pd.DataFrame.from_dict(data_call_arguments,orient='index').to_csv(meta_path / 'data_call_arguments.csv',header=False)

if __name__ == "__main__":
    
    paramsets = ['prestim','poststim']
    regions = ['SCs','SCm','MRN']

    for region in regions: 
        for param in paramsets:
            prproc_and_save(brain_region=region, paramset_name=param, subject_set='active', recompute_data_selection=False)