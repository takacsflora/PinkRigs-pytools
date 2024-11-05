#%%%
from dataset.query import load_data
from dataset.query import queryCSV

exp_kwargs = {
    'subject': ['AV030'],
    'expDate': 'postImplant',
    }

data_name_dict = { 'events': {'_av_trials': 'table'}}

# if you want ephys data
ephys_dict = {'spikes':'all','clusters':'all'}
# both probes 
ephys_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 
data_name_dict.update(ephys_dict)

# camera data
cameras = ['frontCam','sideCam','eyeCam']
cam_dict = {cam:{'camera':['times','ROIMotionEnergy']} for cam in cameras}
data_name_dict.update(cam_dict)
recordings = load_data(data_name_dict = data_name_dict,
                             unwrap_probes= False,
                             merge_probes=True,
                             filter_unique_shank_positions = False,
                             region_selection={'region_name':'MRN',
                                                'framework':'Beryl',
                                                'min_fraction':20,
                                                'goodOnly':True,
                                                'min_spike_num':300},
                            **exp_kwargs
                             )

# %%
