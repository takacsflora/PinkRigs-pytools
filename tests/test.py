#%%%

from pinkrigs_tools.dataset.query import load_data

recordings = load_data(
    subject  = 'AV034',
    expDate = '2022-12-07',
    expDef = 'multiSpaceWorld',
    data_name_dict = 'all-default', 
    merge_probes = True, 
    cam_hierarchy = ['sideCam','frontCam','eyeCam']
)



# %%
ev  = recordings.iloc[0].events._av_trials

import numpy as np
ev.is_validTrial = np.ones(ev.is_auditoryTrial.size)

# %%
from pinkrigs_tools.utils.ev_utils import parse_av_events

ee = parse_av_events(
    ev = ev, 
    contrasts=[0.1,0.2,0.4],
    spls=[0.1],
    vis_azimuths=[-60,60],
    aud_azimuths=[-60,0,60],
    rt_params=None,
    include_unisensory_vis=False,
    include_unisensory_aud=True,
    classify_choice_types = True,
    add_crossval_idx_per_class = True,
    min_trial = 1   
)



# %%
# test ccCP
from pinkrigs_tools.utils.stat.cccp import cccp, get_default_set
pars = get_default_set(which='single_bin',t_length=0.2,t_bin=0.005)
# loading


c = cccp()
c.load_and_format_data(rec=recordings.iloc[0])

# c.aud_azimuths=[0]


u,p,_,t, = zip(*[c.get_U(which_dat='neural',**cp) for _,cp in pars.iterrows()])

# %%
