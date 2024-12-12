import numpy as np
import itertools
import pandas as pd 

from floras_helpers.io import Bunch

def format_events(ev, reverse_opto=False):
        """
        The goal of this function is to process events data and generate/refune features that are commonly used. More sepecifically, this function makes sur : 
        - that both contrast and spl values are rounded to two decimals
        - calculates reaction times values 
        - can flip the data for unilatearl uptogenetics. 
        """
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


def parse_av_events(ev,contrasts,spls,vis_azimuths,aud_azimuths,
                classify_choice_types=True,choice_types = None, 
                min_trial = 2, 
                include_unisensory_aud = True, 
                include_unisensory_vis = False,add_crossval_idx_per_class=False,**kwargs):
        """
        function that preselects categorises events in the av decision-making classs to various classes for stratified cross-validation and cccp.
          1) filters events that we don't intend to fit at all (e.g.using filter_active_trials)
          2) identifies events into sub-categories such that 
            a.) during balanced cross-validation we are able to allocate trialtypes into both train & test sets
            b.) we are able to equalise how many of each of these trial type go into the model at all -- it is e.g. unfair to fill the model with a lot of correct choices and few incorrect choices when wanting to fit 'choice'     

            
        Default exclusions: 
                -  invalid trials
                -  when mouse has a change of mind
        Optional exclusions: 
                - RT is not within range defined by rt_params

        Parameters: 
        classify_choice_types: bool
                - whether to split trials by choice type
        choice_type: list
                which choices to split to 
        rt_params: None/dict
                if dict, it must contain keys 'rt_min' and 'rt_max'
                defines ranges of rt bsed on which a trial is included or not
        min_trial: int
            if we rebalance across trial types this is the minimum trial no. we require from each trial type.   
        include_unisensory_aud: bool
        include_unisensory_vis: bool
        add_crossval_idx_per_class: bool
            whether to split each trial type individially to training and test set

        """

        # keep trials or not based on global criteria 

        ev_ = ev.copy()

        to_keep_trials = ev.is_validTrial.astype('bool')

        print('%.0f trials in total.' % ev.is_auditoryTrial.size)

        print('keeping %.0f valid trials' % to_keep_trials.sum())

        # if it is active data, also exclude trials where firstmove was made prior to choiceMove
        if hasattr(ev,'timeline_firstMoveOn'):
                to_keep_trials = filter_active_trials(ev,**kwargs)


         # and if there is anything else wrong with the trial, like the pd did not get detected..? 

         # and if loose rebalancing strategy is called for choices i.e. something that just factors bias away 

        ev  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})


        ##### TRIAL SORTING #########
        # create the global criteria 
        if classify_choice_types & hasattr(ev,'timeline_choiceMoveDir'):
            # active
            ev.choiceType = ev.timeline_choiceMoveDir
            ev.choiceType[np.isnan(ev.choiceType)] = 0
            if not choice_types:
                choice_types = np.unique(ev.choiceType) 

        else:
            # passive (animal did not go)
            ev.choiceType = np.zeros(ev.is_validTrial.size).astype('int')
            choice_types = [0]
        

        if include_unisensory_aud:                 
            # in unisensory cases the spl/contrast is set to 0 as expected, however the azimuths are also set to nan -- not the easiest to query...
            ev.stim_visAzimuth[np.isnan(ev.stim_visAzimuth.astype('float'))] =-1000 

            vis_azimuths.append(-1000)
            vis_azimuths.sort()
            
            contrasts.append(0)
            contrasts.sort()

            
        
        if include_unisensory_vis:
            ev.stim_audAzimuth[np.isnan(ev.stim_audAzimuth.astype('float'))] =-1000

            aud_azimuths.append(-1000)
            aud_azimuths.sort()

            spls.append(0)
            spls.sort() 
            
        # separate trials into trial classes
        trial_classes = {}
        for idx,(c,spl,v_azi,a_azi,choice) in enumerate(itertools.product(contrasts,spls,vis_azimuths,aud_azimuths,choice_types)):
            # create a dict

            is_this_trial = ((ev.stim_visContrast == c) &
                            (ev.stim_audAmplitude == spl) &
                            (ev.stim_visAzimuth == v_azi) & 
                            (ev.stim_audAzimuth == a_azi) & 
                            (ev.choiceType == choice))
            

            trial_classes[idx] =Bunch ({'is_this_trial':is_this_trial,
                                        'contrast':c,
                                        'spl':spl,
                                        'vis_azimuths':v_azi,
                                        'aud_azimuth':a_azi,
                                         'choice_type': choice,
                                        'n_trials':is_this_trial.sum()})
            
        
            
        # check how balanced the data is....
        #print('attempting to rebalance trials ...')
        trial_class_IDs  = np.array(list(trial_classes.keys()))
        n_in_class = np.array([trial_classes[idx].n_trials for idx in trial_class_IDs])

        # some requested classes don't actually need to be fitted ...
        #print('%.0d/%0d requested trial types have 0 trials in it...' % ((n_in_class==0).sum(),len(trial_classes)))


        min_test  = ((n_in_class>0) & (n_in_class<min_trial))

        if min_test.sum()>0:
            print('some types do not pass the minimum trial requirement. Lower min requirement or pass more data.')
            #trial_class_IDs[min_test]

        # allocate each trialType to train/test set   
        kept_trial_class_IDs = trial_class_IDs[(n_in_class>=min_trial)]
        trial_classes = Bunch({k:trial_classes[k] for k in kept_trial_class_IDs})

        if add_crossval_idx_per_class:
        # divide each trial type into cv sets
                for k in kept_trial_class_IDs: 
                        curr_trial_idx = np.where(trial_classes[k].is_this_trial)[0]

                        np.random.seed(0)
                        np.random.shuffle(curr_trial_idx)
                        middle_index = curr_trial_idx.size//2 
                        train_idx = curr_trial_idx[:middle_index]
                        test_idx = curr_trial_idx[middle_index:] 
                        cv_inds = np.empty(trial_classes[k].is_this_trial.size) * np.nan

                        cv_inds[train_idx] = 1 
                        cv_inds[test_idx] =  2

                        #train=(cv_inds[train_idx,:]*1).sum(axis=0).astype('int')
                        #test =(cv_inds[test_idx,:]*2).sum(axis=0).astype('int')
                        trial_classes[k]['cv_inds'] = cv_inds
       
        # pass on events to the kernels        

        trial_classes = Bunch(trial_classes)
                       
        to_keep_trials = np.sum(np.array([trial_classes[idx].is_this_trial for idx in kept_trial_class_IDs]),axis=0).astype('bool')
        
        # add the trial index type
        ev.trial_type_IDs = np.sum(np.concatenate([((trial_classes[idx].is_this_trial)*(ct))[np.newaxis,:] for ct,idx in enumerate(kept_trial_class_IDs)]),axis=0)

        if add_crossval_idx_per_class: 
                ev.cv_set = np.nansum(np.array([trial_classes[idx].cv_inds for idx in kept_trial_class_IDs]),axis=0) # no. of trials allocated to each class can be still slightly uneven if the trial numbers are odd

        ev  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})

        #print('%.0d trials/%.0d trials are kept.' % (ev.is_validTrial.sum(),ev_.is_validTrial.sum()))

        # recalculate kernel_trialOn and kernel_trialOff, taking into account some all the things that possibly go into the kernel
        # get the df to annotate the trial classes in long form
        contrasts_,spls_,vis_azimuths_,aud_azimuths_,choice_types_,n_trials_=  zip(*[
        (trial_classes[idx].contrast,
                trial_classes[idx].spl,
                trial_classes[idx].vis_azimuths,
                trial_classes[idx].aud_azimuth,
                trial_classes[idx].choice_type,
                trial_classes[idx].n_trials,
                ) 
        for idx in kept_trial_class_IDs])
        
        class_types = pd.DataFrame({
        'contrast': contrasts_, 
        'spl':spls_, 
        'vis_azimuths':vis_azimuths_,
        'aud_azimuths':aud_azimuths_,
        'choice_type':choice_types_,
        'n_trials_':n_trials_
        })



        return ev,class_types
