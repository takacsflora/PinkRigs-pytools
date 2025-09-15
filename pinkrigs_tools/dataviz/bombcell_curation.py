#%%

# load the cluster data from the new bombcell sorting
# this script tests different parameters of bombcell...
# try loading and visualising some additonal parameters...
from pinkrigs_tools.dataset.query import load_data
import numpy as np


import matplotlib.pyplot as plt
from pinkrigs_tools.utils.spk_utils import bombcell_sort_units_new

from pathlib import Path
import pandas as pd

# define parameters of your query
exp_kwargs = {
    'subject': ['AV030'],
    'expDate': '2022-12-09',
    }

ephys_dict = {'spikes':'all','clusters':'all'}
# both probes 
data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 
recordings = load_data(data_name_dict=data_name_dict,**exp_kwargs)

spikes_active = recordings.iloc[0]['probe0']['spikes']
spikes_passive = recordings.iloc[1]['probe0']['spikes']


clusters = recordings.iloc[0]['probe0']['clusters']


#%%


bc_ID,bombcell_class = bombcell_sort_units_new(clusters,max_refractory_period_violations = .5,min_spike_num=1500,min_amp=30)



# %%
bc_curation_df = pd.DataFrame({
    'bc_label': bombcell_class,
    'clusterID': clusters.clusterID
})


#%%

def plot_cluster_summary(cID, axes, additional_params = None):
    cidx = np.where(clusters.clusterID == cID)[0][0]
    ISIs_active = np.diff(np.sort(spikes_active.times[spikes_active.clusters == cID]))
    ISIs_passive = np.diff(np.sort(spikes_passive.times[spikes_passive.clusters == cID]))
    ISIs = np.concatenate((ISIs_active, ISIs_passive))

    # # 1. Matshow waveform
    # ax0 = axes[0]
    # ax0.matshow(
    #     clusters.waveforms[cidx, :, :].T, aspect='auto', cmap='coolwarm_r'
    # )
    # ax0.axis('off')
    # ax0.text(0, 0, f'clusterID: {cID}', color='k', fontsize=10, va='top', ha='left', transform=ax0.transAxes)

    # 2. Lineplot waveforms
    ax1 = axes[0]
    ax1.plot(clusters.waveforms[cidx, :, 0], 'k')
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax1.text(-.1, .5, f'{int(cID)}', color='k', fontsize=7, va='top', ha='left', transform=ax1.transAxes)

    # 3. ISI histogram
    ax2 = axes[1]
    ax2.hist(ISIs[ISIs < .01], bins=100, color='b', alpha=0.7)
    ax2.hist(-ISIs[ISIs < .01], bins=100, color='b', alpha=0.7)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')

    # 4. Scatter active
    ax3 = axes[2]
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.scatter(
        spikes_active.times[spikes_active.clusters == cID],
        spikes_active.amps[spikes_active.clusters == cID],
        edgecolor=None, alpha=0.1, s=.01,color='k'
    )

    # 5. Scatter passive
    ax4 = axes[3]
    ax4.set_xticks([])
    ax4.set_yticks([])
    for spine in ax4.spines.values():
        spine.set_visible(False)
    ax4.scatter(
        spikes_passive.times[spikes_passive.clusters == cID],
        spikes_passive.amps[spikes_passive.clusters == cID],
        edgecolor=None, alpha=0.1, s=.01,color='k'
    )
    if additional_params is not None:
        string = ',\n '.join(['%s: %.2f' % (p,clusters[p][cidx]) for p in additional_params])
        ax4.text(4, 1, string, color='k', fontsize=7, va='top', ha='left', transform=ax1.transAxes)


#%%



cIDs = bc_curation_df[
  #(bc_curation_df['bc_label_old'] == 'good') &
  (bc_curation_df['bc_label_new'] == 'good')
]['clusterID'].values

cIDs = np.random.choice(cIDs,size=min(20,len(cIDs)),replace=False)
#%%

# cIDs = np.round(cIDs) #[53, 56, 60]

# cIDs = [79,74,132,107,168,146,147,93,159,157,158,127]
#%
fig, axs = plt.subplots(len(cIDs), 4, figsize=(2.5, .5 * len(cIDs)), dpi=150)
plt.subplots_adjust(hspace=0, wspace=0)


# 'nSpikes','rawAmplitude'
if len(cIDs) == 1:
    axs = axs[None, :]  # Ensure 2D shape for single row
for i, cID in enumerate(cIDs):
    plot_cluster_summary(cID, axs[i],additional_params=None)
plt.show()



# %%
