# this code queries and extracts av data, 
# more specifically will output a folder of csvs where rows are trials and columns are 
# ev information as well as extracted neural activity


#    


import numpy as np
import pandas as pd
from pathlib import Path

from pinkrigs_tools.dataset.pre_cured import call_
from pinkrigs_tools.utils.ev_utils import format_events,get_triggered_spikes,get_triggered_cam
from pinkrigs_tools.utils.spk_utils import format_cluster_data

from floras_helpers.io import Bunch,save_dict_to_json


def get_trigger_params(paramset='choice'):
    """_summary_

    Args:
        paramset (str, optional): _description_. Defaults to 'choice'.

    Returns:
        _type_: _description_
    """
    if paramset == 'choice':

        timing_params = {
            'onset_time':'timeline_choiceMoveOn',
            'pre_time':2,
            'post_time':.2, 
            'bin_size': 0.01,
        }

    elif paramset == 'stim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':1,
            'post_time':1.7,
            'bin_size': 0.01,
        }

    elif paramset == 'prestim':
        timing_params = {
            'onset_time':'first_stim_time',
            'pre_time':1,
            'post_time':1.7,
            'bin_size': 0.01,
        }
    
    return timing_params

def generate_meta_data(rec_list):
    """
    could be better aha

    """

    rec = rec_list[0] 
    passive_rec = rec_list[1]


    return(pd.DataFrame.from_dict({
            'subject': [rec.subject],
            'expDate': [rec.expDate],
            'rigName': [rec.rigName],
            'passive_expFolder': [passive_rec.expFolder],
            'active_expFolder': [rec.expFolder]
        }))

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

def combine_cam(cam_list):
    cam_combined = {}
    for key in cam_list[0].keys():
        cam_combined[key] = np.concatenate([cam[key] for cam in cam_list])

    return pd.DataFrame.from_dict(cam_combined)
##
def init_folder_structure(savepath,raster_dat = ['stim','choice']):

    trial_path = savepath / 'trial_data'
    cluster_path  = savepath / 'cluster_data'
    spikes_path = savepath / 'spikes_data'
    meta_path = savepath / 'meta_data'
    raster_path = savepath / 'raster_data'
    time_path = savepath / 'time_data'  

    trial_path = Path(trial_path)
    trial_path.mkdir(parents=False,exist_ok=True)

    cluster_path = Path(cluster_path)
    cluster_path.mkdir(parents=False,exist_ok=True)

    spikes_path = Path(spikes_path)
    spikes_path.mkdir(parents=False,exist_ok=True)

    meta_path = Path(meta_path)
    meta_path.mkdir(parents=False,exist_ok=True)

    for raster in raster_dat:
        subfolder = raster_path / raster
        subfolder.mkdir(parents=True, exist_ok=True)


    raster_path = Path(raster_path)
    raster_path.mkdir(parents=False,exist_ok=True)

    time_path = Path(time_path)
    time_path.mkdir(parents=False,exist_ok=True)

    paths = {
        'trials': trial_path,
        'clusters': cluster_path,
        'spikes': spikes_path,
        'meta': meta_path,
        'raster': raster_path,
        'time': time_path
    }

    for raster in raster_dat:
        paths[f'raster_{raster}'] = raster_path / raster


    return paths

def init_folder_structureV2(savepath): 
    data_path = savepath / 'data'
    meta_path = savepath / 'meta_info'

    data_path.mkdir(parents=True,exist_ok=True)
    meta_path.mkdir(parents=True,exist_ok=True)

    paths = {
        'data': data_path,
        'meta': meta_path
    }


    return paths

def combine_sessions(session_list):

    clusters_list = [s.probe.clusters for s in session_list]
    spikes_list = [s.probe.spikes for s in session_list]
    ev_list  = [s.events._av_trials for s in session_list]
    cam_list = [s.camera for s in session_list]

    spikes_list,ev_list,cam_list = cumulate_timings(spikes_list,ev_list,cam_list)

    # cluster data (clusters x cluster features)
    cluster_data = extract_cluster_data(clusters_list)    
    # spikes data (spikes x spike features)
    spikes_data = combine_spikes(spikes_list) # maybe I won't save this one, in the end..

    # cam data
    cam_data = combine_cam(cam_list)

    # event data (trials x trial features)
    ev = pd.concat([format_events(ev,reverse_opto=False) for ev in ev_list])

    return spikes_data,cluster_data,ev,cam_data

def preproc_and_save(brain_region=None,
                    subject_set = 'active',
                    recompute_data_selection = False):
    
    """
    for AV data to basically extract trial aligned neural and movement data
    """

    savepath = Path(r'D:\AV_Neural_Data_Sept2025')


    paths = init_folder_structureV2(savepath)

    data_call_arguments = {
        'subject_set': subject_set,
        'spikeToInclde': True,
        'camToInclude': True,
        'camPCsToInclude': False,
        'recompute_data_selection': recompute_data_selection,
        'unwrap_probes': False,
        'merge_probes': True,
        'filter_unique_shank_positions': False,
        'extra_identifier': f'{brain_region}_',
        'region_selection': {
            'region_name': brain_region,
            'framework': 'Beryl',
            'min_fraction': 12,
            'goodOnly': True,
        },
        'min_rt': 0,
        'analysis_folder': paths['meta'] / 'datasets',	
    }



    active_sessions  = call_(dataset_type = 'active', **data_call_arguments)
    passive_sessions  = call_(dataset_type = 'postactive', **data_call_arguments) # this is the main bottleneck in memory.... 
    trigger_paramsets = ['stim','choice','prestim']

    trigger_timing_params = {t:get_trigger_params(paramset=t) for t in trigger_paramsets} 
    
    for trigger_ts in trigger_paramsets:
        save_dict_to_json(trigger_timing_params,paths['meta']  / f'{trigger_ts}_{brain_region}_trigger_timing_params.json')

    pd.DataFrame.from_dict(data_call_arguments,orient='index').to_csv(paths['meta'] / f'{brain_region}_data_call_arguments.csv',header=False)

    # collect metadata about extraction
    meta_data = []

    # extract the triggered data
    for _,rec in active_sessions.iterrows():
        sessname = '{subject}_{expDate}'.format(**rec) # will be the matching data file
        
        session_path = paths['data'] / sessname
        session_path.mkdir(parents=True,exist_ok=True)

        # find the corresponding passive session
        passive_rec_query = passive_sessions.query('(subject == @rec.subject) & (expDate == @rec.expDate)')
        
        if passive_rec_query.shape[0] == 0:
            print('no passive session found for %s' % sessname)
        
        else:
            print('processing %s' % sessname)
            passive_rec = passive_rec_query.iloc[0]

            spikes_data,cluster_data,ev,cam_data = combine_sessions([rec,passive_rec])
            clusIDs = cluster_data._av_IDs.values

            # triggered data (trials x features[neurons,movement] x timepoints)

            for trigger_ts in trigger_paramsets:
            
                stim_params = trigger_timing_params[trigger_ts]
                R_stim = get_triggered_spikes(
                                    ev=ev,
                                    spikes= spikes_data,
                                    nID = clusIDs,
                                    smoothing = 0, # at this stage I don't want smoothing, can smooth later
                                    **stim_params)
                
                R_cam = get_triggered_cam(
                                    ev=ev,
                                    cam= cam_data,
                                    **stim_params)

                np.savez(session_path / f'raster_{trigger_ts}_aligned.npz', **R_stim)
                np.savez(session_path / f'cam_{trigger_ts}_aligned.npz', **R_cam) 

                # Delete R_stim and R_cam from memory as they can be too large to extract two...
                del R_stim
                del R_cam



            cluster_data.to_csv((session_path / 'clusters.csv'),index=False)
            ev.to_csv((session_path / 'trials.csv'),index=False)         
            meta_data.append(generate_meta_data([rec,passive_rec]))
            print(len(meta_data))




    meta_df = pd.concat(meta_data)

    meta_df.to_csv(paths['meta'] / f'{brain_region}_sessInfo.csv',index=False)
    

if __name__ == "__main__":
    
    #regions = ['SCs','SCm','MOs']

    # regions = ['SCm','SCs']
    # subjects = ['AV005','AV008','AV014','AV025','AV030','AV034','FT030','FT031','FT032']

    # regions = ['MOs']
    # subjects = ['AV007','AV013','AV023']

    regions = ['SCm']
    subjects = ['FT030']
    for subject in subjects:
        for region in regions: 
            try: 
                preproc_and_save(brain_region=region, subject_set=subject, recompute_data_selection=True)
            except Exception as e:
                print(f"Probably no recordings for {subject} in {region}: {e}")
