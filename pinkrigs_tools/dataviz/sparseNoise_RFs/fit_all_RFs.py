#%%

from pathlib import Path
from pinkrigs_tools.dataset.query import load_data
from pinkrigs_tools.utils.stats.rf_model import rf_model
from pinkrigs_tools.utils.spk_utils import format_cluster_data
#
def fit_and_save_rf(rec,
                    savepath=Path(r'D:\AV_Neural_Data_Sept2025\sparseNoise_results')):
    """
    fit the receptive field model and save out all the cluster data with the RF results to a csv file
    rec: recording object, must contain spikes, clusters and _av_trials in events
    savepath: path to save the results to
    """
    clusters  = format_cluster_data(rec.probe.clusters)
    spikes = rec.probe.spikes
    ev = rec.events._av_trials

    m = rf_model(ev = ev,
            spikes = spikes)

    m.fit_evaluate(mode = 'per-neuron') 

    
    df =  m.score.sel(cv_number=1).to_pandas()
    df = df.to_frame(name='rf_VE_test')
    df.reset_index(inplace=True)
    df[['rf_azimuth', 'rf_elevation', 'rf_azimuth_sigma', 'rf_elevation_sigma']] = list(zip(*m.get_rf_degs_from_fit()))

    clusters['neuronID'] = clusters._av_IDs
    clusters = clusters.merge(df,on='neuronID',how='left')

    # save out clusters with RF results
    shanks = clusters._av_shankID.unique()
    shank_str = ''.join(shanks.astype(str))
    probe_folder_stub = '{subject}_{probeID}'.format(**rec) + '_shank' + shank_str
    date_stub = '{expDate}'.format(**rec)
    folder_path = savepath / probe_folder_stub / date_stub
    folder_path.mkdir(parents=True,exist_ok=True)
    clusters.to_csv(folder_path / 'clusters_RF.csv') 


def fit_all_sparse_noise_RFs(subject='AV030'):
    """
    # fit all sparse noise RFs for a given subject
    """

    exp_kwargs = {
        'subject': [subject],
        'expDate': 'all',
        'expDef': ['sparseNoise'],
        'checkEvents':'1',
        'checkSpikes':'1'

        }

    # determine what data to load 
    ephys_dict = {'spikes':'all','clusters':'all'}
    # both probes 
    dat_kwargs = {'events': {'_av_trials': 'all'},'probe0':ephys_dict,'probe1':ephys_dict} 

    recordings = load_data(data_name_dict=dat_kwargs,
                        merge_probes=False,
                        unwrap_probes=True, # we want to keep each probe as a separate recording of thy were recorded together
                        **exp_kwargs)

    for _,rec in recordings.iterrows():
        print(f"Fitting RF for {rec.subject} {rec.expDate} {rec.probeID}")
        fit_and_save_rf(rec)


if __name__ == '__main__':
    fit_all_sparse_noise_RFs(subject='FT035')
# %%
