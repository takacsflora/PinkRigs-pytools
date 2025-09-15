

"""
Functions that deal with querying and loading data from the PinkRig Pipeline
This Pipeline contains metadata in CSVs which we can query using the queryCSV function
 

"""
import json
import re
from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd

# helper functions from my generic packages
from floras_helpers.io import Bunch 

# local functions within library

from ..utils.search import check_date_selection,select_best_camera
from ..utils.spk_utils import is_rec_in_region

def get_csv_location(which):
    """
    function to get the main csv or individual csvs sotring metadata in the PinkRig database
    # maybe we will punlish the metadata csvs? 

    """
    server = Path(r'\\znas.cortexlab.net\Code\PinkRigs')
    if 'main' in which:
        SHEET_ID = '1_hcuWF68PaUFeDBvXAJE0eUmyD9WSQlWtuEqdEuQcVg'
        SHEET_NAME = 'Sheet1'
        csvpath = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    elif 'ibl_queue' in which:
        csvpath = server / r'Helpers/ibl_formatting_queue.csv'
    elif 'pyKS_queue' in which:
        csvpath = server / r'Helpers/pykilosort_queue.csv'
    elif 'training_email' in which:
        csvpath = server / r'Helpers\AVrigEmail.txt'
    else:
        csvpath = server / ('SubjectCSVs/%s.csv' % which)

    return csvpath

def get_server_list():
    """get the list of servers we are currently searching on in CortexLab

    Returns:
        list: pathlib.Path of servers
    """
    server_list = [
        Path(r'\\zinu.cortexlab.net\Subjects'),
        Path(r'\\zaru.cortexlab.net\Subjects'),
        Path(r'\\znas.cortexlab.net\Subjects'),
        Path(r'\\zortex.cortexlab.net\Subjects')
    ]

    return server_list

def queryCSV(subject='all', expDate='all', expDef='all', expNum=None, checkIsSortedPyKS=None, checkEvents=None,
             checkSpikes=None, checkFrontCam=None, checkSideCam=None, checkEyeCam=None):
    """ 
    python version to query experiments based on csvs produced on PinkRigs

    Parameters: 
    ----
    subject : str/list
        selected mice. Can be all, active, or specific subject names
    expDate : str/list
        selected dates. If str: Can be all,lastX,date range, or a single date
                        If list: string of selected dates (e.g. ['2022-03-15','2022-03-30'])
    expDef : str
        selected expdef or portion of the string of the the expdef name

    expNum: str/list 
        selected expNum
    checkIsSortedPyKS: None/str    
        if '2' only outputs

    checkEvents: None\str
        returns match to string if not none ('1', or '2')  
    checkSpikes: None/str
        returns match to string if not none ('1', or '2')  

    check_curation: bool
        only applies if unwrap_independent_probes is True
        whether the data has been curated in phy or not.

    Returns: 
    ----
    exp2checkList : pandas DataFrame 
        concatenated csv of requested experiments and its params 
    """

    mainCSVLoc = get_csv_location('main')
    mouseList = pd.read_csv(mainCSVLoc)
    # look for selected mice
    if 'allActive' in subject:
        mouse2checkList = mouseList[mouseList.IsActive == 1]['Subject']
    elif 'all' in subject:
        mouse2checkList = mouseList.Subject
    else:
        if not isinstance(subject, list):
            subject = [subject]
        mouse2checkList = mouseList[mouseList.Subject.isin(subject)]['Subject']

    exp2checkList = []
    for mm in mouse2checkList:
        mouse_csv = get_csv_location(mm)
        if mouse_csv.is_file():
            expList = pd.read_csv(mouse_csv, dtype='str')
            expList.expDate = [date.replace('_', '-').lower() for date in expList.expDate.values]
            if 'all' not in expDef:
                if not isinstance(expDef, list):
                    expDef = [expDef]

                is_selected_defs = np.concatenate(
                    [expList.expDef.str.contains(curr_expDef).values[np.newaxis] for curr_expDef in expDef]).sum(
                    axis=0).astype('bool')
                expList = expList[is_selected_defs]

            if 'all' not in expDate:
                # dealing with the call of posImplant based on the main csv. Otherwise one is able to use any date they would like 
                if ('postImplant' in expDate):
                    implant_date = mouseList[mouseList.Subject == mm].P0_implantDate
                    # check whether mouse was implanted at all or not.
                    if ~implant_date.isnull().values[0] & (expList.size > 0):
                        implant_date = implant_date.values[0]
                        implant_date = implant_date.replace('_', '-').lower()
                        implant_date_range = implant_date + ':' + expList.expDate.iloc[-1]
                        selected_dates = check_date_selection(implant_date_range, expList.expDate)
                    else:
                        print('%s was not implanted or did not have the requested type of exps.' % mm)
                        selected_dates = np.zeros(expList.expDate.size).astype('bool')

                    expList = expList[selected_dates]

                elif ('last' in expDate):
                    # this only selects the last experiment done on the given animal
                    how_many_days = int(expDate.split('last')[1])
                    expList = expList.iloc[-how_many_days:]

                else:
                    selected_dates = check_date_selection(expDate, expList.expDate)
                    expList = expList[selected_dates]

            if expNum:
                expNum = (np.array(expNum)).astype('str')
                _, idx, _ = np.intersect1d(expList.expNum.to_numpy(), expNum, return_indices=True)
                expList = expList.iloc[idx]

                # add mouse name to list
            expList['subject'] = mm

            exp2checkList.append(expList)

    if len(exp2checkList) > 0:
        exp2checkList = pd.concat(exp2checkList)
        # re-index
        exp2checkList = exp2checkList.reset_index(drop=True)

    else:
        print('you did not call any experiments.')
        exp2checkList = None

    if len(exp2checkList) > 0:
        if checkIsSortedPyKS is not None:
            # nan means we should not have ephys. So we drop nan columns
            exp2checkList = exp2checkList[exp2checkList['issortedPyKS'].notna()]
            to_keep_column = np.array([checkIsSortedPyKS in rec.issortedPyKS for _, rec in exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

        if checkEvents is not None:
            exp2checkList = exp2checkList[exp2checkList['extractEvents'].notna()]
            to_keep_column = np.array([checkEvents in rec.extractEvents for _, rec in exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

        if checkSpikes is not None:
            exp2checkList = exp2checkList[exp2checkList['extractSpikes'].notna()]
            to_keep_column = np.array([checkSpikes in rec.extractSpikes for _, rec in exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

        if checkFrontCam is not None:
            exp2checkList = exp2checkList[
                exp2checkList['alignFrontCam'].notna() & exp2checkList['fMapFrontCam'].notna()]
            to_keep_column = np.array(
                [(checkFrontCam in rec.alignFrontCam) & (checkFrontCam in rec.fMapFrontCam) for _, rec in
                 exp2checkList.iterrows()])
            exp2checkList = exp2checkList[to_keep_column]

    return exp2checkList

def load_ONE_object(collection_folder, object, attributes='all', hacknewbombcellrun = False):
    """
    function that loads any ONE object with npy extension
    ONE object = clollection_folder/object.attribute.expIDtag.extension 
    where extension is either .npy files or parquet table. 

    Parameters
    ----------
    collection_folder: pathlib.Path
    object: str 
        object name: e.g. spikes/clusters/_av_trials
    attributes: str/list
        if str: 'all': all attributes for the object 
        if list: list of strings with specified attributes 
    
    Returns: 
    ---------
    Bunch
        of  object.attribute

    """

    file_names = list(collection_folder.glob('%s.*' % object))
    object_names = [re.split(r"\.", file.name)[0] for file in file_names]
    attribute_names = [re.split(r"\.", file.name)[1] for file in file_names]
    extensions = [re.split(r"\.", file.name)[-1] for file in file_names]

    ## atm I am testing whether to rerun bommcell on the whole dataset or not -- the temp bombcell files are saved in a local folder on my computer, not in the collection folder
    # for now, I will replace the the bombcell paths to the temp paths if it exisits...
    if (hacknewbombcellrun) & (object =='clusters'):
        orig_bc_path = [f for f in file_names if 'bc_qualityMetrics' in  f.stem]
        if len(orig_bc_path) != 0:
            orig_bc_filename = orig_bc_path[0].stem
            # Extract expDate, expNum, subject, probeID, serialNumber from orig_bc_filename
            match = re.match(
                r"clusters\._bc_qualityMetrics\.(?P<expDate>[^_]+)_(?P<expNum>[^_]+)_(?P<subject>[^_]+)_(?P<probeID>[^-]+)-(?P<serialNumber>.+)",
                orig_bc_filename
            )
            expDate = match.group("expDate")
            expNum = match.group("expNum")
            subject = match.group("subject")
            probeID = match.group("probeID")
            serialNumber = match.group("serialNumber")

            ## new ONEs ###
            new_bc_path = rf'D:\bombcell_testrun\ONEs\{subject}\{expDate}\{expNum}\{probeID}\clusters._bc_qualityMetrics.newBC_test.pqt'
    
            if Path(new_bc_path).is_file():
                # replace the original file with the new one
                file_names = [Path(new_bc_path) if f == orig_bc_path[0] else f for f in file_names]


    if 'all' in attributes:
        attributes = attribute_names

    output = {}

    for f, o, a, e in zip(file_names, object_names, attribute_names, extensions):
        # normally don't load any large data with this loader 
        if 'npy' in e:
            if a in attributes:
                tempload = np.load(f)
                if (tempload.ndim == 2):
                    if (tempload.shape[1] == 1):  # if its the stupid matlab format, ravel
                        output[a] = tempload[:, 0]
                    else:
                        output[a] = tempload
                else:
                    output[a] = tempload

        elif 'pqt' in e:
            if a in attributes:  # now I just not load the largeData
                tempload = pd.read_parquet(f)
                tempload = tempload.to_dict('list')

                for k in tempload.keys():
                    if type(tempload[k][0]) == np.ndarray:
                        output[k] = np.array(tempload[k],dtype='object')
                    else: 
                        output[k] = np.array(tempload[k])

        elif 'json' in e:
            if a in attributes:
                tempload = open(f, )
                tempload = json.load(tempload)
                tempload = Path(tempload)
                output[a] = tempload

    output = Bunch(output)

    return output

def load_data(recordings=None,
              data_name_dict=None,
              unwrap_probes=False,
              merge_probes=False,
              region_selection=None,
              filter_unique_shank_positions=False,
              cam_hierarchy=None, **kwargs):
    """
    Paramters: 
    -------------
    recrordings: pd.df
        csv that contains the list of recordings to load.
        In particular the csv ought to contain the column "expFolder" such that points to the parent folder of the ONE_preproc
        if None, then function uses the queryCSVs
         
    data_name_dict: str/dict
        if str: specific default dictionaries can be called, not implemented!
            'all'
            'ev_spk'
            'ev_cam_spk'

        if dict: nested dict that contains requested data
            {collection:{object:attribute}}
            
            note: 
            the raw ibl_format folder can also be called for spiking. 
            For this, one needs to give 'probe0_raw' or 'probe1_raw' as the collection namestring. 
    
    unwrap_probes: bool
        returns exp2checkList with a probe tag where so each row is separate probe as opposed to a session 
    merge_probes: bool
        returns a exp2checkList with a single column of ehys data, where data from both probes are merged into a single Bunch of clusters and spikes
        where clusterIDs from the 2nd probe get 10k added to them

    Returns: 
    -------------
    pd.DataFrame 
        collections  requested by data_name_dict are added as columns  to the original csvs.   

    Todo: implement cond loading,default params
        
    """
    if recordings is None:
        recordings = queryCSV(**kwargs)
    else:
        recordings = recordings[['subject', 'expDate', 'expDef', 'expFolder']]

    recordings = recordings.copy() # set a copy to avoid pandas view versus copy warnings
    if data_name_dict:
        
        # we also provide options to load in all the data default in PinkRigs
        if data_name_dict=='all-default':
            data_name_dict = { 'events': {'_av_trials': 'table'}}
            ephys_dict = {'spikes':'all','clusters':'all'}
            # both probes
            ephys_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 
            data_name_dict.update(ephys_dict)
            # camera data
            cameras = ['frontCam','sideCam','eyeCam']
            cam_dict = {cam:{'camera':['times','ROIMotionEnergy']} for cam in cameras}
            data_name_dict.update(cam_dict)

        collections = list(data_name_dict.keys())
        for collection in collections:
            recordings[collection] = None
            for idx, rec in recordings.iterrows():
                # to do -- make it dependent on whether the extraction was done correctly 

                exp_folder = Path(rec.expFolder)
                if 'raw' in collection and 'probe' in collection:
                    probe_name = re.findall('probe\d', collection)[0]
                    probe_collection_folder = exp_folder / 'ONE_preproc' / probe_name
                    raw_path = list((probe_collection_folder).glob('_av_rawephys.path.*.json'))
                    if len(raw_path) == 1:
                        # then a matching ephys file is found
                        ev_collection_folder = open(raw_path[0], )
                        ev_collection_folder = json.load(ev_collection_folder)
                        ev_collection_folder = Path(ev_collection_folder)
                    else:
                        ev_collection_folder = probe_collection_folder
                else:
                    ev_collection_folder = exp_folder / 'ONE_preproc' / collection

                objects = {}
                for my_object in data_name_dict[collection]:
                    objects[my_object] = load_ONE_object(ev_collection_folder, my_object,
                                                      attributes=data_name_dict[collection][my_object],hacknewbombcellrun=True)
                    


                objects = Bunch(objects)
                recordings.loc[idx][collection] = objects

        ###### we rerun_bombcell temporarily so I might need to add this one here ####


        ### dealing with camera data selection ####
        if cam_hierarchy:
            camdat = [select_best_camera(rec, cam_hierarchy=cam_hierarchy) for _, rec in recordings.iterrows()]
            recordings['camera'] = camdat
            recordings = recordings[~recordings.camera.isna()]

        ### ####### deling with extra arguments that further format the data ######
        # merge probes
        # an optional argument for when there are numerous datasets available for probes, we just merge the data

        if unwrap_probes or merge_probes:
            expected_probe_no = ((recordings.extractSpikes.str.len() / 2) + 0.5)
            expected_probe_no[np.isnan(expected_probe_no)] = 0
            expected_probe_no = expected_probe_no.astype(int)

            recordings['expected_probe_no'] = expected_probe_no

            old_columns = recordings.columns.values
            keep_columns = np.setdiff1d(old_columns, ['ephysPathProbe0', 'ephysPathProbe1', 'probe0', 'probe1'])

            if unwrap_probes & (~merge_probes):
                # this is the mode when we crate a different row for each probe
                rec_list = []
                for _, rec in recordings.iterrows():
                    for p_no in range(rec.expected_probe_no):
                        string_idx = (p_no) * 2
                        if (rec.extractSpikes[int(string_idx)] == '1'):
                            myrec = rec[keep_columns]
                            myrec['probeID'] = 'probe%s' % p_no
                            myrec['probe'] = rec['probe%s' % p_no]
                            ephysPath = rec['ephysPathProbe%s' % p_no]
                            myrec['ephysPath'] = ephysPath

                            curated_fileMark = Path(ephysPath)
                            curated_fileMark = curated_fileMark / 'pyKS\output\cluster_info.tsv'
                            myrec['is_curated'] = curated_fileMark.is_file()

                            rec_list.append(myrec)

                    recordings = pd.DataFrame(rec_list, columns=np.concatenate(
                        (keep_columns, ['probeID', 'probe', 'ephysPath', 'is_curated'])))

                if filter_unique_shank_positions:
                    # only do this with chronic insertions
                    botrow_positions = np.arange(8) * 720
                    botrow_targets = [
                        botrow_positions[np.argmin(np.abs(botrow_positions - min(rec.probe.clusters.depths)))] for
                        _, rec in recordings.iterrows()]
                    recordings['botrow'] = botrow_targets

                    acute_recs = recordings[recordings.rigName == 'lilrig-stim']
                    chronic_recs = recordings[recordings.rigName != 'lilrig-stim']

                    chronic_recs = chronic_recs.drop_duplicates(subset=['subject', 'probeID', 'botrow'])
                    recordings = pd.concat((chronic_recs, acute_recs))



            elif (~unwrap_probes) & merge_probes:
                rec_list = []

                for _, rec in recordings.iterrows():
                    myrec = rec[keep_columns]

                    is_probe0 = hasattr(rec.probe0.spikes, 'amps')
                    is_probe1 = hasattr(rec.probe1.spikes, 'amps')

                    # add new data about probe ID to both spikes and clusters 
                    if is_probe0:
                        rec.probe0.spikes['probeID'] = np.zeros(rec.probe0.spikes.amps.size)
                        rec.probe0.clusters['probeID'] = np.zeros(rec.probe0.clusters.amps.size)

                    if is_probe1:
                        rec.probe1.spikes['probeID'] = np.ones(rec.probe1.spikes.amps.size)
                        rec.probe1.clusters['probeID'] = np.ones(rec.probe1.clusters.amps.size)

                    # redo the _av_IDs of probe 1 if needed
                    if is_probe0 & is_probe1:
                        rec.probe1.clusters._av_IDs = rec.probe1.clusters._av_IDs + 1000
                        rec.probe1.spikes.clusters = rec.probe1.spikes.clusters + 1000

                        sp, cl = rec.probe0.spikes.keys(), rec.probe0.clusters.keys()

                        new_spikes = Bunch(
                            {k: np.concatenate((rec.probe0.spikes[k], rec.probe1.spikes[k])) for k in sp})
                        new_clusters = Bunch(
                            {k: np.concatenate((rec.probe0.clusters[k], rec.probe1.clusters[k])) for k in cl})

                        myrec['probe'] = Bunch({'spikes': new_spikes, 'clusters': new_clusters})
                    elif is_probe0 & ~is_probe1:
                        myrec['probe'] = rec.probe0
                    elif ~is_probe0 & is_probe1:
                        myrec['probe'] = rec.probe1

                    rec_list.append(myrec)

                recordings = pd.DataFrame(rec_list, columns=np.concatenate((keep_columns, ['probe'])))

                # first we identify whether there are two probes
                # then we process the 2nd probes data such that it can be disambiguated from the first probe
                # process the 1st probe
                # merge

            if region_selection is not None:
                keep_rec_region = [is_rec_in_region(rec, **region_selection) for _, rec in recordings.iterrows()]
                recordings = recordings[keep_rec_region]

        # give each recording a unique ID

    return recordings



