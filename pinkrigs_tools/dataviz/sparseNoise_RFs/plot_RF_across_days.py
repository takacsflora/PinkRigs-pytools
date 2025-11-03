#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from floras_helpers.plotting import off_axes

plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})
#%%
def read_rf_csv(f):
    df = pd.read_csv(f)
    df['date'] = f.parent.name
    ID = f.parents[1].stem.split('_')
    df['subject'] = ID[0]
    df['probeID'] = ID[1]
    df['shankID'] = ID[2]
    return df

savepath = Path(r'D:\AV_Neural_Data_Sept2025\sparseNoise_results')

# load all csvs in the folder and its subfolders
all_csvs = list(savepath.rglob('clusters_RF.csv'))
df = pd.concat([read_rf_csv(f) for f in all_csvs])
dfs = df.reset_index()
#%%
df['is_RF'] = df['rf_VE_test'] > 0.05
df['ap_gauss'] = df['ap'] + np.random.normal(0, 50, size=len(df))
df['ap_bregma'] = -df['ap_gauss'] + 5400
 
df['first_day'] = df.groupby(['subject'])['date'].transform('min')
df['days_since_first'] = (pd.to_datetime(df['date']) - 
                          pd.to_datetime(df['first_day'])).dt.days

df['probe_x_shank'] = df['probeID'] + '_shank' + df['_av_shankID'].astype(str)

# good_df = df[(df.bombcell_class!='noise') & 
#              (df.subject=='AV008') & 
#              (df.probeID=='probe1')].copy()

good_df = df[(df.bombcell_class!='noise') & (df.BerylAcronym.isin(['SCs','SCm']))].copy()

RFs = good_df[good_df['is_RF']].copy().reset_index()


#%%
fig, ax = plt.subplots(1,1,dpi=150, figsize=(1,1))
#
RFs['preferred_azimuth'] = RFs['rf_azimuth'] * RFs.hemi

sns.histplot(RFs, x='ap_bregma',y='preferred_azimuth', bins=(50,60),ax=ax)
#sns.scatterplot(data=RFs, x='ap_gauss',y='preferred_azimuth', color='b',s=2,ax=ax,alpha=0.3)
ax.set_ylim([-20,130])
#ax.axhline(60, color='k', linestyle=':', linewidth=0.7)
ax.axhline(45, color='k', linestyle='--', linewidth=0.7)
ax.axhline(75, color='k', linestyle='--', linewidth=0.7)


ax.invert_xaxis()
#%%
fig, ax = plt.subplots(1,1,dpi=150, figsize=(2,2))

sns.histplot(RFs, x='rf_azimuth_sigma',y='dv', bins=(100,50),ax=ax)
#sns.scatterplot(data=RFs, x='rf_azimuth_sigma',y='dv', color='b',s=2,ax=ax,alpha=0.3)
ax.set_ylim([1000,2500])
ax.set_xlim(-5,25)
ax.invert_yaxis()
#%%

# get the mean dir per probe x shank


mean_dirs = RFs.groupby('probe_x_shank')['rf_azimuth'].mean()

mean_dirs = mean_dirs.reset_index().rename(columns={'rf_azimuth':'mean_rf_azimuth'})


# plot mean rf_azimuth per probe x shank on polar coordinates

# colors in coolwarm colormap according to mean dir
cmap = plt.get_cmap('coolwarm')
norm = plt.Normalize(-75, 75)
colors = [cmap(norm(value)) for value in mean_dirs['mean_rf_azimuth']]

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'polar'},
                        figsize=(1,1),dpi=150)
fig.patch.set_alpha(0.0)

# 
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
for i, row in mean_dirs.iterrows():
    mean_dir = np.deg2rad(row['mean_rf_azimuth'])
    color = colors[i]
    ax.annotate('', xy=(mean_dir, 1), xytext=(0,0),
                arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='->', lw=2))
    # ax.text(mean_dir, 1.05, f'{row["mean_rf_azimuth"]:.0f}°', color='k',
    #         fontsize=6, ha='center', va='bottom',rotation=45)
# only show half the plot
ax.set_thetamax(90)
ax.set_thetamin(-90)
ax.set_yticks([])   
ax.set_xticks([])
# Add dashed lines at 0, -90, and 90 degrees
for angle in [0, -90, 90]:
    ax.plot([np.deg2rad(angle), np.deg2rad(angle)], [0, 1], linestyle=':', color='black', linewidth=.7,alpha=0.7)

stim_left = np.arange(-75,-45,1)

stim_right = np.arange(45,75,1)

ax.fill_between(np.deg2rad(stim_left), 0, 1, color='blue', alpha=0.1)
ax.fill_between(np.deg2rad(stim_right), 0, 1, color='red', alpha=0.1)

ax.plot(np.deg2rad(stim_left), np.ones_like(stim_left), color='k', alpha=0.3, linewidth=3)
ax.plot(np.deg2rad(stim_right), np.ones_like(stim_right), color='k', alpha=0.3, linewidth=3)
ax.axis('off')



#%%





#%%

mean_RF_depth = RFs.depths.mean()
#80 and 20 percentiles of depths
depth_20 = RFs.depths.quantile(0.2)
depth_80 = RFs.depths.quantile(0.8)

print(depth_20, mean_RF_depth, depth_80)



#%%
# in a half polar plot, plot the histogram of rf azimuths for each _av_shankID

# actually azimuths are already in degrees, if they are negative then the stim os on the left and positive then the stim is on the right. 
# 
# only plot half of the polar plot

plt.rcParams.update({'font.size': 6})
shankIDs = good_df['_av_shankID'].unique()
n_shanks = len(shankIDs)
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'polar'},
                        figsize=(1.2*n_shanks,1.4),dpi=150)

shank_colors = sns.color_palette('tab10',n_shanks)


fig.subplots_adjust(wspace=0.3, hspace=0.3)
for i, shankID in enumerate(shankIDs):
    shank_RFs = RFs[(RFs['_av_shankID']==shankID)].copy()
    azimuths = shank_RFs['rf_azimuth'].values
    # convert to radians
    azimuths_rad = np.deg2rad(azimuths)  
    # histogram
    bins = np.deg2rad(np.arange(-110, 111,5))
    counts, _ = np.histogram(azimuths_rad, bins=bins)
    # plot histogram as bar plot
    widths = np.diff(bins)
    # ax.bar(bins[:-1], counts, width=widths, bottom=0.0, align='edge', color=shank_colors[i], edgecolor='k',linewidth=.01, alpha=0.7)

    # # plot the mean direction
    mean_dir = np.deg2rad(shank_RFs['rf_azimuth'].mean())
    # put a text in the color of the shank
    ax.text(mean_dir, 1.1, f'{shank_RFs["rf_azimuth"].mean():.0f}°', color=shank_colors[i],
            fontsize=6, ha='center', va='bottom')
    
    # put an arrow towards the mean dir, all the same size
    ax.annotate('', xy=(mean_dir, 1), xytext=(0,0),
                arrowprops=dict(facecolor=shank_colors[i], edgecolor=shank_colors[i], arrowstyle='->', lw=1))


    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)


    # only show half the plot
    ax.set_thetamax(110)
    ax.set_thetamin(-110)
    ax.set_yticks([])


#%%
# plot all subject RFs across days
plt.rcParams.update({'font.size': 6})

days = np.sort(good_df['days_since_first'].unique())

#days = [0,6,13,20,28]
n_days = len(days)
fig, axs = plt.subplots(1, int(np.ceil(n_days/1)), 
                        figsize=(.5*n_days,1.5),sharey=True,dpi=150)


axs = axs.flatten()

fig.subplots_adjust(wspace=-0.1, hspace=-0.05)
for i, day in enumerate(days):
    ax = axs[i]
    day_nrns = good_df[good_df['days_since_first'] == day].copy()
    day_RFs = day_nrns[day_nrns['is_RF']].copy()
    sns.stripplot(data = day_nrns,x = '_av_shankID',color='grey',
                  y='depths',size=2, ax=ax,alpha=0.2)
    
    sns.stripplot(data=day_RFs,x='_av_shankID',y='depths',
                  hue='rf_azimuth', palette='coolwarm',
                  size=6,edgecolor='k',linewidth=0.5,
                hue_norm=(-110,110),
                  alpha=0.8,
                  ax=ax,legend=False)
    
    ax.set_title(f'{day}')
    off_axes(ax)
    ax.set_ylim([depth_20-500,depth_80+500])
# Remove unused axes
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

# %%

# let's do the same plot as above but with the anatomy plotter 


from floras_helpers.anat_plots import anatomy_plotter

anat = anatomy_plotter()
coord  = 1100
days = [0,11]#5,15,22]
n_days = len(days)

fig, axs = plt.subplots(1, int(np.ceil(n_days/1)), 
                        figsize=(1.2*n_days,1.2),sharey=True,dpi=150)


axs = axs.flatten()

fig.subplots_adjust(wspace=0.05, hspace=-0.05)
for i, day in enumerate(days):
    ax = axs[i]
    day_nrns = good_df[good_df['days_since_first'] == day].copy()
    day_RFs = day_nrns[day_nrns['is_RF']].copy()

    coord = day_nrns['ml'].median()-5600

    anat.plot_anat_canvas(ax=ax,coord = coord, axis='ml')

    anat.plot_points(day_nrns['ap'],day_nrns['dv'],unilateral=True,c = 'grey',alpha=0.2,marker = '.',
                     s=10,edgecolor=None)
    
    anat.plot_points(day_RFs['ap'],day_RFs['dv'],unilateral=True,
                     c=day_RFs['rf_azimuth'],alpha=0.8,
                     marker = '.',s=70,edgecolor='k',linewidth=0.5,
                     cmap='coolwarm',vmin=-75,vmax=75)
    # ax.set_xlim([-2000,2000])
    
    ax.set_xlim([-2700,-4750])
    ax.set_ylim([-2700,0])
    off_axes(ax)


# %%
# do the same along the ml axis 