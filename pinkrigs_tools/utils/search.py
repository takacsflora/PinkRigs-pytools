
import datetime
import numpy as np
import re
import os 

def check_date_selection(date_selection, date_list):
    """
   funct to match a called date range to a list of dates (indicating all epxeriments in the csv, for example)

    Parameters:
    -----------
    date selection: If str: Can be all,lastX,date range, or a single date
                    If list: string of selected dates (e.g. ['2022-03-15','2022-03-30'])
        corresponds to the selected dates
         
    date_list: list
        corresponds to all dates to match to 
    
    Return: list
        list of dates selected from dateList that pass the criteria determined by date_selection
    """
    date_range = []
    date_range_called = False  # when a from to type of date range called. Otherwise date_selection is treated as list of dates
    if 'previous' in date_selection:
        date_range_called = True
        date_selection = date_selection.split('previous')[1]
        date_range.append(datetime.datetime.today() - datetime.timedelta(days=int(date_selection)))
        date_range.append(datetime.datetime.today())
    else:
        if type(date_selection) is not list:
            # here the data selection becomes a list anyway
            date_range_called = True
            date_selection = date_selection.split(':')

        for d in date_selection:
            date_range.append(datetime.datetime.strptime(d, '%Y-%m-%d'))
            # if only one element
        if len(date_range) == 1:
            date_range.append(date_range[0])

    selected_dates = []
    for date in date_list:
        exp_date = datetime.datetime.strptime(date, '%Y-%m-%d')

        # check for the dates
        if date_range_called:
            IsGoodDate = (exp_date >= date_range[0]) & (exp_date <= date_range[1])
        else:
            IsGoodDate = True in ([exp_date == date_range[i] for i in range(len(date_range))])

        if IsGoodDate:
            selected_dates.append(True)
        else:
            selected_dates.append(False)
    return selected_dates

def select_best_camera(rec, cam_hierarchy=['sideCam', 'frontCam', 'eyeCam']):
    """
    helper function that select a camera data based on the hierarchy and based on whether the camera data is avlaible at all 

    Parameters: 
    -----------
    rec: pd.Series 
        loaded ONE object 
    cam_hierarchy: list
        determines which camera view are prioritised

    """
    cam_checks = np.array([(hasattr(rec[cam].camera, 'ROIMotionEnergy') &
                            hasattr(rec[cam].camera, 'times')) if hasattr(rec, cam) else False for cam in
                           cam_hierarchy])
    if cam_checks.any():
        cam_idx = np.where(cam_checks)[0][0]
        used_camname = cam_hierarchy[cam_idx]
        cam = rec[used_camname].camera
    else:
        cam = None

    if hasattr(cam, 'ROIMotionEnergy'):
        if cam.ROIMotionEnergy.ndim == 2:
            cam.ROIMotionEnergy = (cam.ROIMotionEnergy[:, 0])

    return cam


def get_latest_file(directory, prefix="result_", suffix=".txt"):
    # Regular expression to extract the timestamp from the filename
    timestamp_pattern = re.compile(r'{}(\d{{4}}-\d{{2}}-\d{{2}}-\d{{6}}){}'.format(prefix, suffix))
    
    latest_file = None
    latest_time = None

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        match = timestamp_pattern.match(filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H%M%S')

            # Compare timestamps to find the latest one
            if latest_time is None or timestamp > latest_time:
                latest_time = timestamp
                latest_file = filename

    return latest_file