# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from floras_helpers.binning import get_binned_rasters

from pinkrigs_tools.dataset.query import load_data
from pinkrigs_tools.utils.ev_utils import format_events
from pinkrigs_tools.utils.stat.cccp import cccp, get_default_set

recordings = load_data(
    subject  = 'AV008',
    expDate = '2022-03-11',
    expDef = 'multiSpaceWorld',
    data_name_dict = 'all-default', 
    merge_probes = True, 
    cam_hierarchy = ['sideCam','frontCam','eyeCam']
)


rec = recordings.iloc[0]
#%%
pars = get_default_set(which='single_bin',t_length=0.2,t_bin=0.005)
c = cccp()
c.load_and_format_data(rec=rec)
u,p,_,t, = zip(*[c.get_U(which_dat='neural',**cp) for _,cp in pars.iterrows()])

p_choice = p[1]
# %%
ev = pd.DataFrame(format_events(rec.events._av_trials))
spikes = rec.probe.spikes
ev = ev.query('response_direction != 0')

# make all the rasters aligned to auditory stimulus onset and choice
# then in the filtering function we just sort for that matrix
raster_kwargs = {
    'spike_times': spikes.times,
    'spike_clusters': spikes.clusters,
    'cluster_ids': rec.probe.clusters._av_IDs, 
    'bin_size':0.005,
    'return_fr': True,
    'baseline_subtract': False
}

at_aud = get_binned_rasters(
    align_times = ev.timeline_audPeriodOn,
    pre_time = 0.2, post_time =0.5,**raster_kwargs
    )

at_choice = get_binned_rasters(
    align_times = ev.timeline_choiceMoveOn, 
    pre_time = 0.15,post_time  = 0.05, **raster_kwargs
)


# %%

plotted_aud_azimuth = np.unique(ev.audDiff)
plotted_vis_azimuth = np.unique(ev.visDiff)
plotted_choice  = np.unique(ev.response_direction)

n_aud = len(plotted_aud_azimuth)
n_vis = len(plotted_vis_azimuth)
n_choice = len(plotted_choice)

assert n_choice==2, 'this plotting function is just for left right choices.'
#%%

def plot_sempsth(responses,t_bin,ax,**kwargs):    
    n_trials = responses.shape[0]
    m = responses.mean(axis=0)
    sem = responses.std(axis=0) / np.sqrt(n_trials)
    ax.plot(t_bin,m,**kwargs)
    ax.fill_between(t_bin, m - sem, m + sem,alpha= 0.3,**kwargs)

neuronID = 1166

nrn_at_aud = (at_aud.rasters[:,np.isin(at_aud.cscale,neuronID),:]).mean(axis=1)
nrn_at_choice = (at_choice.rasters[:,np.isin(at_choice.cscale,neuronID),:]).mean(axis=1)

# is_choice = (p_choice < 0.05)[:,0]
# nrn_at_aud = (at_aud.rasters[:,is_choice,:]).mean(axis=1)
# nrn_at_choice = (at_choice.rasters[:,is_choice,:]).mean(axis=1)


# nnn = [np.where(is_choice)[0][4]]
# nrn_at_aud = (at_aud.rasters[:,nnn,:]).mean(axis=1)
# nrn_at_choice = (at_choice.rasters[:,nnn,:]).mean(axis=1)


cazi,aazi,vazi=np.meshgrid(plotted_choice,plotted_aud_azimuth,plotted_vis_azimuth)
fig,ax = plt.subplots(n_choice,n_aud*2,figsize=(n_aud*6,n_choice*3),sharey=True)
cmap  =getattr(plt.cm,'coolwarm')
colors = cmap(np.linspace(0,1,n_vis))

for aidx,a in enumerate(plotted_aud_azimuth):
    for cidx,c in enumerate(plotted_choice):
        for v,vcolor in zip(plotted_vis_azimuth,colors):
            trial_idx = np.where((ev.visDiff == v) & (ev.audDiff==a) & (ev.response_direction==c))[0]
            if len(trial_idx>0):
                plot_sempsth(nrn_at_aud[trial_idx,:],at_aud.tscale, ax = ax[cidx,aidx*2],color = vcolor) 
                plot_sempsth(nrn_at_choice[trial_idx,:],at_choice.tscale, ax = ax[cidx,aidx*2+1],color = vcolor) 

plt.show()

# %%
