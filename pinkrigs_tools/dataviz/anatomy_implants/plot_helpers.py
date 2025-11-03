import numpy as np
import matplotlib.pyplot as plt



def get_region_colors():
    allen_dict = {"SCs":"#04D9FF",
            "SCm":"#FF04D9",
            "MOs":"#7BD1A3",
            "VISp":"#C0C9CF",

        }
    
    allen_dict['Vis'] = allen_dict['VISp']
    allen_dict['SC'] = allen_dict['SCm']
    allen_dict['Frontal'] = allen_dict['MOs']
    
    return allen_dict


def rgb_to_hex(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r,g,b)


def brainrender_scattermap(values,n_bins=None,vmin=None,vmax= None,cmap='viridis'):
    """
    function that produces the colors for brainrender scatter 
    Parameters: 
    -----------
    values: np.ndarray 
        values that determine the hue of the scatter 
    cmap: str
        colormap by matplotlib
    vmin: None/float
    vmax: None/float
        values to determine colormap edges

    Returns: list of strings
    --------
        hex map of for each value in values 
    """

    if vmin is None:
        vmin = min(values)

    if vmax is None: 
        vmax = max(values)

    
    map_min,map_max = 0,1
    if n_bins is None: 
        n_bins = int(np.round(values.size/15,0))        
    my_cbins = np.linspace(map_min,map_max,int(n_bins+1))
    my_datbins = np.linspace(vmin,vmax,n_bins)
    color_func = plt.cm.get_cmap(cmap)
    colors_ = color_func(my_cbins)
    colors_ = [rgb_to_hex((c[:3]*255).astype('int')) for c in colors_]
    cbin_idxs = np.digitize(values,bins=my_datbins) 

    values_cmap = [colors_[i] for i in cbin_idxs]
    

    return values_cmap

