# this code queries and extracts av data, 
# more specifically will output a folder of csvs where rows are trials and columns are 
# ev information as well as extracted neural activity
# maybe this collection should also contain the design matrix formatter..? up to decision for later

#    


import numpy as np
import pandas as pd
from pathlib import Path

from pinkrigs_tools.dataset.pre_cured import call_
from pinkrigs_tools.utils.ev_utils import format_events,get_triggered_data
from pinkrigs_tools.utils.spk_utils import format_cluster_data

from floras_helpers.io import Bunch,save_dict_to_json

# for fitting the rt distributions 
from scipy.special import erf
import scipy.optimize as opt

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
            'pre_time':0.5,
            'post_time':.2, 
            'bin_size': 0.02,
        }

    elif paramset == 'prestim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0.15,
            'post_time':0, 
            'bin_size': 0.02,
        }
    
    elif paramset == 'poststim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0,
            'post_time':0.15,
            'bin_size': 0.02,
        }

    elif paramset == 'stim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0.2,
            'post_time':0.5,
            'bin_size': 0.02,
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

def exgaussian_pdf(x, mu, sigma, tau):
    return (1 / tau) * np.exp((mu - x) / tau + (sigma ** 2) / (2 * tau ** 2)) * \
           0.5 * (1 + erf((x - mu - (sigma ** 2) / tau) / (np.sqrt(2) * sigma)))

# Negative log-likelihood for better peak fitting
def neg_log_likelihood_exgaussian(params,reaction_times):
    mu, sigma, tau = params
    if sigma <= 0 or tau <= 0:  # Prevent invalid values
        return np.inf
    return -np.sum(np.log(exgaussian_pdf(reaction_times, mu, sigma, tau) + 1e-8))  # Avoid log(0)

def fit_exgaussian(reaction_times):
    initial_guess = [np.mean(reaction_times), np.std(reaction_times), np.mean(reaction_times) - np.median(reaction_times)]
    result = opt.minimize(neg_log_likelihood_exgaussian, initial_guess, args=(reaction_times,), method='Nelder-Mead')
    return result.x 

def sample_exgaussian(mu, sigma, tau, size=10000):
    gaussian_part = np.random.normal(mu, sigma, size)
    exponential_part = np.random.exponential(tau, size)
    return gaussian_part + exponential_part  # Sum of Gaussian + Exponentia

def add_choice_onset_to_passive_trials(ev_active,ev_passive):
    
    # get the choice onset times from the active session relative to auditory stimulus onset
    reaction_times = ev_active.rt_aud[~np.isnan(ev_active.rt_aud)]
    mu, sigma, tau = fit_exgaussian(reaction_times)

    # yeah I think I will quicky just fit the RT distribution for each stimulus condition and then sample from that distribution.
    n_passive_trials = ev_passive.is_auditoryTrial.size

    resampled_rts = sample_exgaussian(mu, sigma, tau, size=n_passive_trials) 

    # for now I will just add this. In the future it might be useful to fit the RT-dist for each stimulus condition/choice 
    # and then resample both the rt and the choice for the passive trials.

    ev_passive['timeline_choiceMoveOn'] = ev_passive.timeline_audPeriodOn + resampled_rts
    return ev_passive

def preproc_and_save(brain_region=None,
                    subject_set = 'active',
                    recompute_data_selection = False):

    savepath = Path(r'D:\AV_Neural_Data') #\%s_%s' % (brain_region,paramset_name))

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
            'min_fraction': 20,
            'goodOnly': True,
            'min_spike_num': 300
        },
        'min_rt': 0,
        'analysis_folder': savepath / 'datasets',	
    }



    active_sessions  = call_(dataset_type = 'active', **data_call_arguments)
    passive_sessions  = call_(dataset_type = 'postactive', **data_call_arguments)


    trigger_paramsets = ['stim','choice']

    paths = init_folder_structure(savepath,raster_dat = trigger_paramsets)
    
    # check parameters of extraction 
    # save parameters of extraction

    trigger_timing_params = {t:get_trigger_params(paramset=t) for t in trigger_paramsets} 
    
    for trigger_ts in trigger_paramsets:
        save_dict_to_json(trigger_timing_params,paths[f'raster_{trigger_ts}']  / f'{brain_region}trigger_timing_params.json')

    pd.DataFrame.from_dict(data_call_arguments,orient='index').to_csv(paths['meta'] / f'{brain_region}_data_call_arguments.csv',header=False)


    # collect metadata about extraction
    meta_data = []

    for _,rec in active_sessions.iterrows():
        sessname = '{subject}_{expDate}.csv'.format(**rec) # will be the matching data file
        
        
        # find the corresponding passive session
        passive_rec_query = passive_sessions.query('(subject == @rec.subject) & (expDate == @rec.expDate)')
        
        if passive_rec_query.shape[0] == 0:
            print('no passive session found for %s' % sessname)
        
        else:
            passive_rec = passive_rec_query.iloc[0]


            # to compare passive activity from approx. stimulus response vs. pre-choice neural activity, I will also simulate the timing of choice onset on passive trials 
            # for this I sample the time of choice onset from the active session and add it to the passive session

            ev_passive = add_choice_onset_to_passive_trials(rec.events._av_trials,passive_rec.events._av_trials)

            # build a list of all the data we are extracting  (btw this is potentailly how ut would work if we concatenate several days)
            clusters_list = [rec.probe.clusters,passive_rec.probe.clusters]
            spikes_list = [rec.probe.spikes,passive_rec.probe.spikes]
            ev_list  = [rec.events._av_trials,ev_passive]
            cam_list = [rec.camera,passive_rec.camera]

            spikes_list,ev_list,cam_list = cumulate_timings(spikes_list,ev_list,cam_list)

            # cluster data (clusters x cluster features)
            cluster_data = extract_cluster_data(clusters_list)
            clusIDs = cluster_data._av_IDs.values
            
            # spikes data (spikes x spike features)
            spikes_data = combine_spikes(spikes_list) # maybe I won't save this one, in the end..

            # cam data
            cam_data = combine_cam(cam_list)

            # get the trigged data 

            # event data (trials x trial features)
            ev = pd.concat([format_events(ev,reverse_opto=False) for ev in ev_list])
            
            
            # triggeed data (trials x features[neurons,movement] x timepoints)

            for trigger_ts in trigger_paramsets:
                triggered_data =get_triggered_data(
                                        ev=ev,
                                        spikes= spikes_data,
                                        nID = clusIDs,cam=cam_data,
                                        single_average_accross_neurons=False,
                                        get_zscored=False,
                                        **trigger_timing_params[trigger_ts]
                                        )
                                    
                # probably save as parquet ?
                triggered_data.to_parquet((paths[f'raster_{trigger_ts}'] / sessname),index=False)

            
            # time data (feautres[neurons,movements]  x timepoints) 
            # this is pass atm but we will do it. 
            



            # Save the data
            #spikes_data.to_csv((paths['spikes'] / sessname),index=False) --maybe no need


            cluster_data.to_csv((paths['clusters'] / sessname),index=False)
            ev.to_csv((paths['trials'] / sessname),index=False)         
            meta_data.append(generate_meta_data([rec,passive_rec]))




    meta_df = pd.concat(meta_data)

    meta_df.to_csv(paths['meta'] / f'{brain_region}_sessInfo.csv',index=False)
    

if __name__ == "__main__":
    
    regions = ['VISp']

    for region in regions: 
        preproc_and_save(brain_region=region, subject_set='active', recompute_data_selection=True)