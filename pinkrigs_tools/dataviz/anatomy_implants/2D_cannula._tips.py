# %%
# takes histology folder of requested animals and plots their canulla location if it exists 
import sys
import numpy as np
import pandas as pd 
from pathlib import Path

from pinkrigs_tools.dataset.query import queryCSV

from floras_helpers.hist.atlas import AllenAtlas,BrainRegions

atlas,br = AllenAtlas(25),BrainRegions()

subjects = ['AV029','AV031','AV033','AV036','AV038','AV041','AV044','AV046','AV047','AV055','AV052','AV053','AV054','AV056','AV057'] # list of subjects that we intent to query 

#subjects = ['AV041']
recordings = queryCSV(subject=subjects,expDate='last1')

stub = r'Histology\registration\brainreg_output\manual_segmentation\standard_space\tracks'
histology_folders = [
    (Path(r.expFolder).parents[1] / stub) for _,r in recordings.iterrows()
]

# %%

# save summary anatomical data: subject,ap,dv,ml,hemisphere(-1:Left,1:Right),regionAcronym 

data = pd.DataFrame()
for idx,m in enumerate(histology_folders):
    cannulae_list = list(m.glob('*.npy'))
    for c in cannulae_list:
        subject = m.parents[5].name
        track = np.load(c)
        # canulla tip point (because I always start tracking at the tip)
        tip_ccf = track[0]
        # assert the position of these tip points in allen atlas space location
        region_id = atlas.get_labels(atlas.ccf2xyz(track[0],ccf_order='apdvml'))
        region_acronym=br.id2acronym(region_id) # get the parent of that 

        data = data.append(
            {'subject':subject,
            'ap':tip_ccf[0], 
            'dv':tip_ccf[1],
            'ml':tip_ccf[2], 
            'hemisphere':-int(np.sign(tip_ccf[2]-5600)), 
            'region_id':region_id, 
            'region_acronym':region_acronym[0],
            'parent1':br.acronym2acronym(region_acronym, mapping='Beryl')[0]},ignore_index=True
        )

# save this as a file
data.to_csv(r'D:\AV_opto_data\cannula_locations.csv')

# %%
