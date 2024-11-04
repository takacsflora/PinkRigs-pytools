import numpy as np

def format_events(ev, reverse_opto=False):
    if hasattr(ev, 'stim_visContrast'):
        ev.stim_visContrast = np.round(ev.stim_visContrast, 2)
    if hasattr(ev, 'stim_audAmplitude'):
        ev.stim_audAmplitude = np.round(ev.stim_audAmplitude, 2)

        amps = np.unique(ev.stim_audAmplitude)

        if (
        amps[amps > 0]).size == 1:  # if there is only one amp then the usual way of calculating audDiff etc is valid
            ev.visDiff = ev.stim_visContrast * np.sign(ev.stim_visAzimuth)
            ev.visDiff[np.isnan(ev.visDiff)] = 0
            ev.audDiff = np.sign(ev.stim_audAzimuth)

    if hasattr(ev, 'timeline_choiceMoveOn'):
        ev.rt = ev.timeline_choiceMoveOn - np.nanmin(
            np.concatenate([ev.timeline_audPeriodOn[:, np.newaxis], ev.timeline_visPeriodOn[:, np.newaxis]], axis=1),
            axis=1)
        ev.rt_aud = ev.timeline_choiceMoveOn - ev.timeline_audPeriodOn
        ev.first_move_time = ev.timeline_firstMoveOn - np.nanmin(
            np.concatenate([ev.timeline_audPeriodOn[:, np.newaxis], ev.timeline_visPeriodOn[:, np.newaxis]], axis=1),
            axis=1)
    if hasattr(ev, 'is_laserTrial') & hasattr(ev, 'stim_laser1_power') & hasattr(ev, 'stim_laser2_power'):
        ev.laser_power = (ev.stim_laser1_power + ev.stim_laser2_power).astype('int')
        ev.laser_power_signed = (ev.laser_power * ev.stim_laserPosition)
        if reverse_opto & ~(np.unique(ev.laser_power_signed > 0).any()):
            # if we call this than if within the session the opto is only on the left then we reverse the azimuth and choices on that session
            ev.stim_audAzimuth = ev.stim_audAzimuth * -1
            ev.stim_visAzimuth = ev.stim_visAzimuth * -1
            ev.timeline_choiceMoveDir = ((ev.timeline_choiceMoveDir - 1.5) * -1) + 1.5

    return ev


def filter_active_trials(ev,rt_params = {'rt_min':None,'rt_max':None},exclude_premature_wheel=False):
        """
        function to filter out typically unused trials in the active data analysis
        """
        ev = format_events(ev)
        to_keep_trials = ((ev.is_validTrial) & 
                (ev.response_direction!=0) &  # not a nogo
                (np.abs(ev.stim_audAzimuth)!=30))
         # if spikes are used we need to filter extra trials, such as changes of Mind
        if exclude_premature_wheel:
                no_premature_wheel = (ev.timeline_firstMoveOn-ev.timeline_choiceMoveOn)==0
                to_keep_trials = to_keep_trials & no_premature_wheel

        if rt_params:
                if rt_params['rt_min']: 
                        to_keep_trials = to_keep_trials & (ev.rt>=rt_params['rt_min'])
                
                if rt_params['rt_max']: 
                        to_keep_trials = to_keep_trials & (ev.rt<=rt_params['rt_max'])  


        return to_keep_trials