# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from floras_helpers.binning import get_binned_rasters
from floras_helpers.plotting import off_axes

from pinkrigs_tools.dataset.query import load_data
from pinkrigs_tools.utils.ev_utils import format_events
from pinkrigs_tools.utils.stat.cccp import cccp, get_default_set

recordings = load_data(
    subject  = 'AV030',
    expDate = '2022-12-08',

    expDef = 'multiSpaceWorld',
    data_name_dict = 'all-default', 
    merge_probes = True, 
    cam_hierarchy = ['sideCam','frontCam','eyeCam']
)


rec = recordings.iloc[0]

ev = rec.events._av_trials
spikes = rec.probe.spikes
#%%
# pars = get_default_set(which='single_bin',t_length=0.2,t_bin=0.005)
# c = cccp()
# c.load_and_format_data(rec=rec)
# u,p,_,t, = zip(*[c.get_U(which_dat='neural',**cp) for _,cp in pars.iterrows()])

# p_choice = p[1]
# %%
ev = pd.DataFrame(format_events(rec.events._av_trials))

spikes = rec.probe.spikes

ev = ev.query('abs(stim_audAzimuth) != 30')
ev = ev.query('response_direction != 0')

#%%

# make all the rasters aligned to auditory stimulus onset and choice
# then in the filtering function we just sort for that matrix
raster_kwargs = {
    'spike_times': spikes.times,
    'spike_clusters': spikes.clusters,
    'cluster_ids': rec.probe.clusters._av_IDs, 
    'bin_size':0.02,
    'return_fr': True,
    'baseline_subtract': False,
    'smoothing':0.1
}

at_aud = get_binned_rasters(
    align_times = ev.timeline_audPeriodOn,
    pre_time = 0.2, post_time =0.5,**raster_kwargs
    )

at_choice = get_binned_rasters(
    align_times = ev.timeline_choiceMoveOn, 
    pre_time = 1.5,post_time  = 0, **raster_kwargs
)


# %%


#%%

def plot_sempsth(responses,t_bin,ax,**kwargs):    
    n_trials = responses.shape[0]
    m = responses.mean(axis=0)
    sem = responses.std(axis=0) / np.sqrt(n_trials)
    ax.plot(t_bin,m,**kwargs)
    ax.fill_between(t_bin, m - sem, m + sem,alpha= 0.3,**kwargs)

def plot_trials(responses,t_bin,ax,**kwargs):
    ax.plot(t_bin,responses.T,**kwargs)

def plot_responses(
        ev,stim_resp,choice_resp,additinal_conds,nID,
        plot_kw = {},
        at_stim = False,
        per_trial = True,
        subplot_scale = 1, 
        ): 
    

    plotted_aud_azimuth = np.unique(ev.audDiff)
    plotted_vis_azimuth = np.unique(ev.visDiff)

    n_aud = len(plotted_aud_azimuth)
    n_vis = len(plotted_vis_azimuth)

    nID = np.where(stim_resp.cscale == nID)

    assert len(nID)==1,'neuron not found'
    nID = nID[0] 

    fig,ax = plt.subplots(n_aud,n_vis,
                    #figsize=(n_vis*subplot_scale,n_aud*subplot_scale),
                    figsize=(n_vis*subplot_scale,n_vis*subplot_scale),
                    sharey=True,sharex=True)


    cmap  =getattr(plt.cm,'coolwarm')
    colors = cmap(np.linspace(0,1,n_aud))

    for aidx,a in enumerate(plotted_aud_azimuth):
        
        for vidx,v in enumerate(plotted_vis_azimuth):
            
            trial_idx = np.where((ev.visDiff == v) & (ev.audDiff==a) & additinal_conds)[0]
            current_color  = colors[aidx]

            current_axis = ax[n_aud-1-aidx, vidx]

            if len(trial_idx>0):
                
                if at_stim:
                    resp = stim_resp.rasters[trial_idx,nID,:]
                    tscale = stim_resp.tscale
                else:
                    resp = choice_resp.rasters[trial_idx,nID,:]
                    tscale = choice_resp.tscale



                plot_kw['color'] = current_color

                if per_trial:
                    plot_trials(resp,tscale,current_axis,**plot_kw)

                else:
                    plot_sempsth(resp,tscale,current_axis,**plot_kw) 

            off_axes(current_axis)
            current_axis.set_facecolor((1, 1, 1, 0))

    plt.subplots_adjust(wspace=0.01, hspace=-.5)

    return fig

def find_axlim(fig):
    global_min, global_max = float('inf'), float('-inf')
    for ax in fig.get_axes():
        for line in ax.get_lines():  # Iterate over all line objects in the Axes
            ydata = line.get_ydata()
            global_min = min(global_min, np.min(ydata))
            global_max = max(global_max, np.max(ydata))
    return global_min,global_max

def plot_several_responses(
        neuronID = 1336,
        divider_arg = 'response_direction',**kwargs):

    allfigs = []
    for ttt in np.sort(ev[divider_arg].unique()):


        trial_indices = ev[divider_arg]==ttt

        fig = plot_responses(
            ev,at_aud,at_choice,trial_indices,neuronID, 
            **kwargs)
        
        allfigs.append(fig)

    mins,maxs =zip(*[find_axlim(f) for f in allfigs])
    global_ylim = (np.array(mins).min(),np.array(maxs).max())  # Define your desired ylim

    for fig in allfigs:
        for ax in fig.get_axes():
            ax.set_ylim(global_ylim)

    plt.show()



plot_several_responses(
    neuronID=95,
    divider_arg='response_direction', 
    at_stim = True, per_trial=False
)
# %%

fig,ax = plt.subplots(1,1,figsize=(5,5))
ax.matshow(at_choice.rasters[:,neuronID,:],aspect='auto',cmap='Greys',vmax = 200)
# %%
nrn_raster = at_choice.rasters[:,neuronID,:]
n_trials = nrn_raster.shape[0]

for i in range(10):

    plt.plot(nrn_raster[i,:])


# %%
#for each trial calculate when was the earliest time that you reached 50% of the final firing rate

def get_half_fr_time(nrn_at_trial):
    fr_at_choice = nrn_at_trial[-1]
    half_fr = fr_at_choice/2
    idx = np.where(nrn_at_trial < half_fr)[0]

    if len(idx) == 0:
        return np.nan
    else:
        return at_choice.tscale[idx[-1]]


half_fr_times = np.array([get_half_fr_time(nrn_raster[i,:]) for i in range(n_trials)])



# %%
evidence = (ev.audDiff + ev.visDiff)

plt.scatter(ev.rt,half_fr_times,c=evidence,cmap='coolwarm')

  # %%


# normalise each trial to the last timebin
nrn_raster_norm = np.zeros_like(nrn_raster) 
for i in range(n_trials):
    nrn_raster[i,:] = nrn_raster[i,:] - nrn_raster[i,-1]



#%%
for i in range(10):

    plt.plot(nrn_raster_norm[i,:])
#plt.plot(np.mean(at_choice.rasters[:,neuronID,:],axis=0))


# %%
t = nrn_raster#_norm
plt.plot(t[ev.is_visualTrial,:].mean(axis=0))
plt.plot(t[ev.is_auditoryTrial,:].mean(axis=0))
plt.plot(t[ev.is_coherentTrial,:].mean(axis=0))
plt.plot(t[ev.is_conflictTrial,:].mean(axis=0))

 # %%



