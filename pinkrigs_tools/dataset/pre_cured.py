"This scrit contains a way to call precurated datasets for the audiovisual multisensory integration task task, currently. The idea is, here you can precompile dataset calling functions."
import datetime
from pathlib import Path 
import numpy as np 
import pandas as pd

from .query import load_data
from ..utils import ev_utils

def call_(subject_set='naive',dataset_type='naive', extra_identifier = '',
                    spikeToInclde=True, 
                    camToInclude = True, camPCsToInclude = False,
                    recompute_data_selection=False,
                    min_rt=0,min_active_trials=150,
                    analysis_folder = None,**kwargs):
    """
    Function to load the neural data in a pre-curated manner (i.e. typical active data, typical passive data etc.). 
    This function is useful for selecting data for analyisis for the audiovisual task.

    Parameters:
    subject_set : str, optional
        Specifies what animals to load. Can be a nickname for a set of animals, or a specific subject
    dataset_type : str, optional
        Specifies what type of experiments to load. Options include :
                'naive', 'all', 'active', 'allPassive','postactive'. Default is 'naive'.
    spikeToInclde : bool, optional
        Whether to include spike data in the query. Default is True.
    camToInclude : bool, optional
        Whether to include camera data in the query. Default is True.
    recompute_data_selection : bool, optional
        Whether to recompute data selection. If False, loads from a saved file. Default is False.
    min_rt : float, optional
        Minimum reaction time for filtering trials in the 'active' dataset. Default is 0.
    analysis_folder : str or Path, optional
        Path to the folder where analysis results should be saved. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the data loading functions.

    Returns:
    pd.DataFrame
        DataFrame containing the selected recordings with relevant data.
    """


    # build up the parameters of the data
    data_name_dict = { 'events': {'_av_trials': 'table'}}
    
    
    if camPCsToInclude:
        cam_details = {'camera':['times','ROIMotionEnergy','_av_motionPCs']}
    else:
        cam_details = {'camera':['times','ROIMotionEnergy']}

    cam_dict = {'frontCam':cam_details,'eyeCam':cam_details,'sideCam':cam_details}
    ephys_dict = {'spikes':'all','clusters':'all'}
    ephys_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 

    if 'naive' in subject_set:
        subject_list = ['FT008','FT009','FT010','FT011','FT019','FT022','FT025','FT027','FT038','FT039']
    elif 'all' in subject_set: 
        subject_list = 'all'
    elif 'active' in subject_set:
        subject_list = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034']
    elif 'forebrain' in subject_set:
        subject_list = ['AV007','AV009','AV013','AV015','AV021','AV023']
    elif 'SCMOs' in subject_set:
        subject_list = ['FT030','FT032','AV005','AV008','AV014','AV025','AV030','AV034',
                        'AV007','AV013','AV023']
    elif 'totAV' in subject_set: 
        subject_list = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034','AV007','AV009','AV013','AV015','AV021','AV023']
    else:
        subject_list = [subject_set]
    

    # this is becase the naive subjects were recorded in lilrig, everything else was in PinkRigs
    if 'naive' in subject_set and camToInclude:
        cam_hierarchy = ['frontCam','eyeCam','sideCam']
    elif 'naive' not in subject_set and camToInclude:
        cam_hierarchy =  ['sideCam','eyeCam','frontCam']
    else: 
        cam_hierarchy = None

    
    if camToInclude:
        data_name_dict.update(cam_dict)

    if spikeToInclde:
        data_name_dict.update(ephys_dict)

    
    query_params = {
        'subject': subject_list,
        'data_name_dict':data_name_dict,
        'checkEvents':'1', 
        'cam_hierarchy': cam_hierarchy
    }

    # in case I want ot call any further parameter in addiiton
    query_params.update(kwargs)

    if spikeToInclde:
        query_params['checkSpikes']='1' 




    # do the query
    if analysis_folder is None:
        savepath = Path(r'D:\data_extractions')
    else:
        savepath = Path(analysis_folder)

    savepath.mkdir(parents=False,exist_ok=True)


    if (dataset_type == 'naive') & recompute_data_selection:   
        recordings = load_data(expDef = ['AVPassive_spatialIntegrationFlora_with_ckeckerboard',
                                        'AVPassive_ckeckerboard_updatechecker',
                                        'AVPassive_ckeckerboard_extended',
                                        'AVPassive_checkerboard_extended'],
                                            **query_params)
        
    elif (dataset_type=='allPassive') & recompute_data_selection:
        recordings = load_data(expDef = ['AVPassive','postactive'],
                                            **query_params)
        
    elif (dataset_type == 'postactive') & recompute_data_selection:
        recordings = load_data(expDef = 'postactive',expDate = 'postImplant', **query_params)

    elif (dataset_type == 'active') & recompute_data_selection:
        recordings = load_data(expDef = 'multiSpaceWorld',expDate = 'postImplant', **query_params)  

        exclude_premature_wheel = False
        rt_params = {'rt_min':min_rt+0.03,'rt_max':1.5}

        n_trials  = np.array([ev_utils.filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()]) # I think this is potentailly creating problems because it overwrites the ev structure permanently?
        recordings = recordings.iloc[n_trials>min_active_trials]  

    # make the savepath
    elif not recompute_data_selection:
        # find the latest file related to this recording  
        datasets = list(savepath.glob(f'{extra_identifier}{subject_set}_{dataset_type}_dataset_*.csv'))
        savefile = datasets[0]
        expList = pd.read_csv(savefile)   
        query_params.pop('subject') # each identifier will be called separetly so we don't need this anymore      

        recordings = [load_data(subject = rec.subject,
                                expDate = rec.expDate,
                                expNum = rec.expNum,
                                **query_params) for _,rec in expList.iterrows()]    
        
        recordings = pd.concat(recordings)

    

    if spikeToInclde:
        # basically double check spiking because the extractSpikes==1 is not enough. Some alignments are weird
        # see e.g. AV025/2022-11-07/4
        is_good_spike = np.array([len(rec.probe.spikes.clusters)>0 for _,rec in recordings.iterrows()])
        recordings = recordings.iloc[is_good_spike] 

    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    filename = f"{extra_identifier}{subject_set}_{dataset_type}_dataset_{current_timestamp}.csv"
    savefile = savepath / filename
    recordings[['subject','expDate','expNum','expFolder']].to_csv(savefile) # I need to timestamp it -- plus arrange into some sort of folder

    return recordings