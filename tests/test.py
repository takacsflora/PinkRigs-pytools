#%%%
from pinkrigs_tools.dataset.query import load_data
from pinkrigs_tools.dataset.query import queryCSV

exp_kwargs = {
    'subject': ['AV030'],
    'expDate': '2022-12-07:2022-12-15',
    'expDef': 'multiSpaceWorld'
    }

recordings = load_data(data_name_dict = 'all-default',
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
