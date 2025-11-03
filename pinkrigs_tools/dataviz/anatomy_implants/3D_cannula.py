

# %%
# show tracks in brainrender
import numpy as np
from pathlib import Path


from pinkrigs_tools.dataset.query import queryCSV
from brainrender import Scene,settings
from brainrender.actors import Points
from plot_helpers import brainrender_scattermap,get_region_colors
# Create a brainrender scene

# Turn off axes before creating your scene
settings.SHADER_STYLE = 'plastic'  # smooth, shaded look
settings.SHOW_AXES = False
settings.EDGE_COLOR = None


subjects = ['AV029','AV031','AV033','AV036','AV038','AV041','AV044','AV046','AV047','AV055'] # list of subjects that we intent to query 
#subjects = ['AV052', 'AV053', 'AV056', 'AV057']
#subjects = ['AV041']
recordings = queryCSV(subject=subjects,expDate='last1')

stub = r'Histology\registration\brainreg_output\manual_segmentation\standard_space\tracks'
histology_folders = [
    (Path(r.expFolder).parents[1] / stub) for _,r in recordings.iterrows()
]
n_mice = len(histology_folders)
mouse_colors = brainrender_scattermap(np.arange(n_mice),vmin=-1,vmax=n_mice+1,cmap='Set1',n_bins=n_mice)

# test opening of files
for idx,m in enumerate(histology_folders):
    cannulae_list = list(m.glob('*.npy'))
    for c in cannulae_list:
        track = np.load(c)
        print(c)
        print(track.shape)

#%%



scene = Scene(title="", inset=False,root=True)

# Add brain regions
brain = scene.root
brain.alpha = 0.01

region_colors = get_region_colors()
scene.add_brain_region("SCs",alpha=0.3,color=region_colors['SCs'])
sc = scene.add_brain_region("SCm",alpha=0.3,color=region_colors['SCm'])
#scene.add_brain_region("MOs",alpha=0.3,color = region_colors['MOs'])
#scene.add_brain_region("VISp",alpha=0.3,color  = region_colors['VISp'] )


# if len(histology_folders)>0:

#     for idx,m in enumerate(histology_folders):
#         cannulae_list = list(m.glob('*.npy'))
#         for c in cannulae_list:
#             track = np.load(c)
#             scene.add(Points(track, colors=mouse_colors[idx], radius=60, alpha=1))
#             scene.add(Points(track[0][np.newaxis,:], colors=mouse_colors[idx], radius=120, alpha=1))


# this is side view
# cam = {
#     "pos": (11654, -32464, 81761),
#     "viewup": (0, -1, -1),
#     "clipping_range": (32024, 63229),
#     "focalPoint": (7319, 2861, -3942),
#     "distance": 43901,
# }

# Update the camera settings to view the object from the top-ish tilted 20 degrees left-to-right
cam = {
    "pos": (75000, -55000, 30000), 
    "viewup": (0, -1, 0),           # Set the upward direction of the camera
    "clipping_range": (32024, 63229),
    "focalPoint": (7319, 2861, -3942),  # Focus on the object
    "distance": 50000,             # Adjust the distance to ensure the object is in view
}


# cam = {
#     "pos": (-36430, 0, 35700),
#     "viewup": (0, -1, -1),
#     "clipping_range": (32024, 63229),
#     "focalPoint": (7319, 2861, -3942),
#     "distance": 43901,
# }

scene.render(zoom=2.0,interactive=False,camera = cam)


savepath = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\cannula_positions')




cpath = savepath / f'SC.png'
scene.screenshot(str(cpath),scale=4)


# interValue = True 
# pltView = 'coronal'
# pltSlice = True
# if pltSlice:
#     scene.slice("frontal")



# if pltView == "coronal":
#     cam = {
#         "pos": (-36430, 0, -5700),
#         "viewup": (0, -1, 0),
#         "clippingRange": (40360, 64977),
#         "focalPoint": (7319, 2861, -3942),
#         "distance": 43901,
#     }
# elif pltView == "side":
#     cam = {
#         "pos": (11654, -32464, 81761),
#         "viewup": (0, -1, -1),
#         "clippingRange": (32024, 63229),
#         "focalPoint": (7319, 2861, -3942),
#         "distance": 43901,
#     }

# scene.render(interactive=interValue,camera=cam,zoom=3.5)