import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timezone
from scipy.stats import t
import scipy.io
from scipy.stats import wilcoxon, ttest_ind, ttest_1samp
import random
import time
from multiprocessing import Process, Manager, Pool
import sys
from itertools import repeat
from sklearn.covariance import LedoitWolf
from sklearn import linear_model
from sklearn.linear_model import RidgeCV


def _create_condition_list(id):
    match id:
        case 0:
            return ['All-conditions']
        case 1:
            return ['Long-dist', 'Short-dist']
        case 2:
            return ['Up-dir', 'Down-dir', 'Left-dir', 'Right-dir']
        case 3:
            return ['Top-pos', 'Bottom-pos', 'Left-pos', 'Right-pos', 'Center-pos']



def _get_combined_condition_idcs(id, markers):
    match id:
        case 0:
            combined_conditions = [m for m in markers]

        case 1:
            longs = [m for m in markers if '-l' in m]
            shorts = [m for m in markers if '-s' in m]
            combined_conditions = [longs, shorts]

        case 2:
            ups = [m for m in markers if 'BTT' in m]
            downs = [m for m in markers if 'TTB' in m]
            lefts = [m for m in markers if 'RTL' in m]
            rights = [m for m in markers if 'LTR' in m]
            combined_conditions = [ups, downs, lefts, rights]

        case 3:
            tops = [m for m in markers if 'BTT-l' in m]
            bots = [m for m in markers if 'TTB-l' in m]
            lefts = [m for m in markers if 'RTL-l' in m]
            rights = [m for m in markers if 'LTR-l' in m]
            centers = [m for m in markers if '-s']
            combined_conditions = [tops, bots, lefts, rights, centers]

    return combined_conditions

def ci_calc(chan, n_ts, avg):
    n_bootstrap = 10000
    n_sbj = len(avg)
    confidence = .95
    ch_ci = np.zeros((2,n_ts)) # First entry is uppers, second entry is lowers
    for ts in range(n_ts):
        vals = []
        for subj in range(n_sbj):
            vals.append(avg[subj][chan, ts])

        # Bootstrapping for confidence interval:
        values = np.random.choice(vals, size=(len(vals), n_bootstrap), replace=True).mean(axis=0)
        # values = [np.random.choice(vals, size=len(vals), replace=True).mean() for i in range(n_bootstrap)]
        ch_ci[1,ts], ch_ci[0,ts] = np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)])
    return ch_ci

def _calc_confidence_interval(avg, bootstrap=True, multiproc=True):
    n_bootstrap = 1000
    n_chan, n_ts = avg[0].shape
    n_sbj = len(avg)
    uppers = np.zeros((n_chan, n_ts))
    lowers = np.zeros((n_chan, n_ts))

    confidence = .95

    start = time.time()
    # for chan in range(n_chan):
    p = Pool(processes=n_chan)
    channel_ci = p.starmap(ci_calc, zip(range(n_chan), repeat(n_ts), repeat(avg)))

    # print(len(channel_ci))
    # Extract ci's:
    for ch in range(n_chan):
        # print(channel_ci[0].shape)
        uppers[ch,:] = channel_ci[ch][0,:]
        lowers[ch,:] = channel_ci[ch][1,:]

    print(f'One condition took me: {round((time.time()-start),2)}') #, end='\r')
    return uppers, lowers


def grand_avg_per_subject(src, dst, sbj_list, split_id=0):
    # split_id: 0...['All together']
    #           1...['long', 'short']
    #           2...['up', 'down', 'left', 'right']
    #           3...['top', 'bottom', 'left', 'right', 'center']

    # Create the list with the conditions based on split_id:
    condition_list = _create_condition_list(split_id)

    # Retrieve all filenames from the source directory:
    file_names = [f for f in os.listdir(src)]

    for sbj in sbj_list:
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]

        # Get alignment:
        alignment = ''
        if 'cue_aligned' in file:
            alignment = 'cue-aligned'
        elif 'movement_aligned' in file:
            alignment = 'movement-aligned'

        # Load epochs file:
        epochs = mne.read_epochs(file, preload=True)

        # Get markers:
        markers = list(epochs.event_id.keys())

        # Create combined conditions based on split_id:
        combined_conditions = _get_combined_condition_idcs(split_id, markers)

        # Append the average activity of each participant shape = (epochs, channels, times):
        # Averaging all epochs for each timestamp and channel:
        for i, cond in enumerate(condition_list):
            avg = epochs[combined_conditions[i]].get_data().mean(axis=0)

            # Store the subject grand averages:
            store_name = f'{dst}/sbj-grand-avg_{sbj}_{alignment}_{cond}.npy'
            np.save(store_name, avg)

def grand_avg(src, dst, split_id, alignment):
    # List all files with naming convention 'sbj-grand_avg_<sbj>_<alignment>_<condition>
    type = 'sbj-grand-avg'

    condition_list = _create_condition_list(split_id)

    for cond in condition_list:
        # Retrieve all filenames from the source directory that contain grand-avg-sbj, conditions and alignment:
        file_names = [f for f in os.listdir(src) if (type in f) and (alignment in f) and (cond in f)]

        sbj_avg = []
        for f in file_names:
            file = src + '/' + f
            # Read all files and combine to grand average:
            sbj_avg.append(np.load(file))
        grand_avg = np.array(sbj_avg).mean(axis=0)

        # Store grand_avg:
        store_name = f'{dst}/grand-avg_{alignment}_{cond}.npy'
        np.save(store_name, grand_avg)

def confidence_interval(src, dst, split_id, alignment):
    # List all files with naming convention 'sbj-grand_avg_<sbj>_<alignment>_<condition>
    type = 'sbj-grand-avg'

    condition_list = _create_condition_list(split_id)

    for cond in condition_list:
        # Retrieve all filenames from the source directory that contain grand-avg-sbj, conditions and alignment:
        file_names = [f for f in os.listdir(src) if (type in f) and (alignment in f) and (cond in f)]

        sbj_avg = []
        for f in file_names:
            file = src + '/' + f
            # Read all files and combine to grand average:
            sbj_avg.append(np.load(file))

        uppers, lowers = _calc_confidence_interval(sbj_avg, bootstrap=True)

        # Store grand_avg:
        store_name = f'{dst}/grand-avg-uppers_{alignment}_{cond}.npy'
        np.save(store_name, uppers)
        store_name = f'{dst}/grand-avg-lowers_{alignment}_{cond}.npy'
        np.save(store_name, lowers)

def statistical_analysis(src, dst, split_id, alignment):
    # List all files with naming convention 'sbj-grand_avg_<sbj>_<alignment>_<condition>
    type = 'sbj-grand-avg'

    condition_list = _create_condition_list(split_id)

    # combinations = _get_condition_combinations(split_id)

    for cond in condition_list:
        # Retrieve all filenames from the source directory that contain grand-avg-sbj, conditions and alignment:
        file_names = [f for f in os.listdir(src) if (type in f) and (alignment in f) and (cond in f)]

        sbj_avg = []
        for f in file_names:
            file = src + '/' + f
            # Read all files and combine to grand average:
            sbj_avg.append(np.load(file))


        # TODO: Implement permutation_tests if necassary
        #p_vals = _perform_permuation_tests(sbj_avg)

        # Store grand_avg:
        # store_name = f'{dst}/grand-avg-uppers_{alignment}_{cond}.npy'
        # np.save(store_name, uppers)
        # store_name = f'{dst}/grand-avg-lowers_{alignment}_{cond}.npy'
        # np.save(store_name, lowers)


def get_timings(src, dst, sbj_list, split_id):
    # split_id: 0...['All together']
    #           1...['long', 'short']
    #           2...['up', 'down', 'left', 'right']
    #           3...['top', 'bottom', 'left', 'right', 'center']

    # Create the list with the conditions based on split_id:
    condition_list = _create_condition_list(split_id)

    # Retrieve all filenames from the source directory:
    file_names = [f for f in os.listdir(src)]

    for sbj in sbj_list:
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]

        # Get alignment:
        alignment = ''
        if 'cue_aligned' in file:
            alignment = 'cue-aligned'
        elif 'movement_aligned' in file:
            alignment = 'movement-aligned'

        # Load epochs file:
        epochs = mne.read_epochs(file, preload=True)

        if split_id == 0:
            cue_times, release_times, touch_times = _get_cue_movement_onsets(epochs.annotations)

            # Store the subject grand averages:
            store_name = f'{dst}/timings-cue_{sbj}_{alignment}.npy'
            np.save(store_name, cue_times)
            store_name = f'{dst}/timings-release_{sbj}_{alignment}.npy'
            np.save(store_name, release_times)
            store_name = f'{dst}/timings-touch_{sbj}_{alignment}.npy'
            np.save(store_name, touch_times)

        elif split_id == 1:
            cue_times_l, release_times_l, touch_times_l = _get_cue_movement_onsets(epochs.annotations, abbr='l')
            cue_times_s, release_times_s, touch_times_s = _get_cue_movement_onsets(epochs.annotations, abbr='s')

            # Store the subject grand averages:
            store_name = f'{dst}/timings-cue-l_{sbj}_{alignment}.npy'
            np.save(store_name, cue_times_l)
            store_name = f'{dst}/timings-release-l_{sbj}_{alignment}.npy'
            np.save(store_name, release_times_l)
            store_name = f'{dst}/timings-touch-l_{sbj}_{alignment}.npy'
            np.save(store_name, touch_times_l)

            # Store the subject grand averages:
            store_name = f'{dst}/timings-cue-s_{sbj}_{alignment}.npy'
            np.save(store_name, cue_times_s)
            store_name = f'{dst}/timings-release-s_{sbj}_{alignment}.npy'
            np.save(store_name, release_times_s)
            store_name = f'{dst}/timings-touch-s_{sbj}_{alignment}.npy'
            np.save(store_name, touch_times_s)


def _get_cue_movement_onsets(annot, abbr=None):

    if abbr == None:
        trial_type_markers = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    elif abbr == 'l':
        trial_type_markers = ['LTR-l', 'RTL-l', 'TTB-l', 'BTT-l']
    elif abbr == 's':
        trial_type_markers = ['LTR-s', 'RTL-s', 'TTB-s', 'BTT-s']

    # Get difference between cue onset and movement onset (*i*1):
    cue_times = []
    release_times = []
    touch_times = []
    for i, entry in enumerate(annot.description):
        if entry in trial_type_markers:
            if 'bad' in annot.description[i+1]:
                continue
            else:
                # Get delay between cue which is 'Cue' at i+3 and ix1 at i+4 and cx0 at i+5
                cue_times.append(annot.onset[i+3])
                release_times.append(annot.onset[i+4])
                touch_times.append(annot.onset[i+5])

    diff_cue_release = np.array(release_times) - np.array(cue_times)
    diff_cue_finished = np.array(touch_times) - np.array(cue_times)
    diff_start_stop = np.array(touch_times) - np.array(release_times)

    #return list(diff_cue_release), list(diff_cue_finished), list(diff_start_stop)
    return np.array(cue_times), np.array(release_times), np.array(touch_times)


def classification_mean_and_ci(src, dst):
    start = time.time()
    # Get mean and confidence interval from classification dataframe:
    df = pd.read_csv(f'{src}/classification_df.csv', index_col=0)

    # Calculate CI for all subjects with the same 'N_timepoints', 'Type' and 'Condition' column for each timepoint.
    cols = ['Timepoint', 'Mean_accuracy', 'Type', 'N_timepoints', 'Condition', 'Upper', 'Lower']
    confidence = .95
    df_mean = pd.DataFrame(columns=cols)
    for type in df['Type'].unique():
        for n_timepoints in df['N_timepoints'].unique():
            for condition in df['Condition'].unique():
                print(f'{type} {condition} {n_timepoints}')
                timepoints = df[(df['Condition']==condition) & (df['N_timepoints']==n_timepoints) & (df['Type']==type)]['Timepoint'].unique()
                for tp in timepoints:
                    # Get average overall subjects for each timepoint:
                    accs = np.array(df[(df['Type'] == type) & (df['N_timepoints'] == n_timepoints) & (df['Condition'] == condition) & (df['Timepoint'] == tp)]['Accuracy'])
                    mean = accs.mean()

                    # Bootstrapping for confidence interval:
                    n_bootstrap = 10000
                    values = np.random.choice(accs, size=(len(accs),n_bootstrap), replace=True).mean(axis=0)
                    # values = [np.random.choice(accs,size=len(accs),replace=True).mean() for i in range(1000)]
                    lower, upper = np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)])

                    df_mean.loc[len(df_mean)] = [tp, mean, type, n_timepoints, condition, upper, lower]

    df_mean = df_mean.dropna()
    df_mean.to_csv(f'{dst}/classification_means_ci.csv')

    print(f'Calculation took me: {time.time() - start}seconds.')


def glm(src, dst, sbj_list, split_id, alignment, shrink=True):

    # Retrieve all filenames from the source directory:
    file_names = [f for f in os.listdir(src)]

    for sbj in sbj_list:
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]

        # Load epochs file:
        epochs = mne.read_epochs(file, preload=True)

        epochs_ids = equalize_epochs(epochs)

        S = _create_parameter_matrix(epochs[epochs_ids], z_scoring=True)

        _,_,A = _fit_glm(S, epochs[epochs_ids], shrinkage=shrink)

        # Save regression coefficients:
        if shrink:
            store_name = f'{dst}/regr-coeff_{sbj}_{alignment}_shrink.npy'
        else:
            store_name = f'{dst}/regr-coeff_{sbj}_{alignment}_no-shrink.npy'
        np.save(store_name, A)

        # Save parameter matrix:
        store_name = f'{dst}/param-matrix_{sbj}_{alignment}.npy'
        np.save(store_name, S)

    np.save(f'{dst}/ch_names.npy', np.array(epochs.ch_names))


def _create_parameter_matrix(epochs, z_scoring=True):
    """
    Creates a parameter matrix for the given epochs.

    :param epochs: A list of MNE-Python Epochs objects.
                   Each Epochs object represents a segment of EEG data.
    :type epochs: list of mne.Epochs objects

    :param z_scoring: Whether to standardize each parameter by subtracting its mean and dividing by its standard deviation.
                      Defaults to True.
    :type z_scoring: bool

    :return: The parameter matrix for the given epochs.
             The matrix has shape (5, N), where N is the number of epochs.
             If `z_scoring` is True, each parameter is standardized by subtracting its mean and dividing by its standard deviation.
             The first row represents the short condition (1 if the marker contains '-s', 0 otherwise).
             The second row represents the long condition (1 if the marker contains '-l', 0 otherwise).
             The third row represents the vertical condition (1 if the marker contains 'BTT' or 'TTB', 0 otherwise).
             The fourth row represents the horizontal condition (1 if the marker contains 'LTR' or 'RTL', 0 otherwise).
             The fifth row represents the intercept (always 1).
    :rtype: numpy.ndarray
    """

    # Create s vectors:
    s_short, s_long, s_vert, s_horz, s_intercept = np.empty((1,len(epochs))), np.empty((1,len(epochs))), np.empty((1,len(epochs))), np.empty((1,len(epochs))), np.ones((1,len(epochs)))
    s_short[:], s_long[:], s_vert[:], s_horz[:] = np.nan, np.nan, np.nan, np.nan

    for epoch in range(len(epochs)):
        marker = list(epochs[epoch].event_id.keys())[0]
        if '-s' in marker:
            s_short[0, epoch] = 1
            s_long[0, epoch] = 0
        elif '-l' in marker:
            s_short[0, epoch] = 0
            s_long[0, epoch] = 1

        if 'BTT' in marker or 'TTB' in marker:
            s_horz[0, epoch] = 0
            s_vert[0, epoch] = 1
        elif 'LTR' in marker or 'RTL' in marker:
            s_horz[0, epoch] = 1
            s_vert[0, epoch] = 0

    # z-score each parameter if the flag is true:
    if z_scoring:
        # Test for z-scoring:
        # s_full = np.concatenate([s_short.squeeze(), s_long.squeeze(), s_vert.squeeze(), s_horz.squeeze(),
        #                          s_intercept.squeeze()])
        # s_short = (s_short - s_full.mean())/s_full.std()
        # s_long = (s_long - s_full.mean())/s_full.std()
        # s_vert = (s_vert - s_full.mean())/s_full.std()
        # s_horz = (s_horz - s_full.mean())/s_full.std()
        # s_intercept = (s_intercept - s_full.mean())/s_full.std()

        s_short = (s_short - s_short.mean())/s_short.std()
        s_long = (s_long - s_long.mean())/s_long.std()
        s_vert = (s_vert - s_vert.mean())/s_vert.std()
        s_horz = (s_horz - s_horz.mean())/s_horz.std()

    # Return S matrix:
    S = np.concatenate((s_short, s_long, s_vert, s_horz, s_intercept),axis=0)
    return S.T

def equalize_epochs(epochs):
    # Equalize epochs:
    markers = list(epochs.event_id.keys())

    trial_type_markers = ['BTT-s', 'BTT-l', 'TTB-s', 'TTB-l', 'RTL-s', 'RTL-l', 'LTR-s', 'LTR-l']
    marker_count = np.zeros((8))
    marker_ids = [[],[],[],[],[],[],[],[]]
    for epoch in range(len(epochs)):
        marker = list(epochs[epoch].event_id.keys())[0][:5]
        marker_position = trial_type_markers.index(marker)
        marker_ids[marker_position].append(epoch)
        marker_count[marker_position] += 1

    # Get minimal count:
    min_count = int(marker_count.min())

    # Randomly sample from the 8 idcs a number of min count samples:
    equalized_ids = []
    for elem in range(len(marker_ids)):
        equalized_ids += list(random.sample(marker_ids[elem], k=min_count))

    return sorted(equalized_ids)

def _fit_glm(S, epochs, shrinkage=False):
    """
    Applies a generalized linear model to estimate the contribution of experimental conditions to EEG data.

    :param S: array, shape (n_conditions, n_trials)
        The matrix of experimental conditions.
    :param epochs: mne.Epochs
        The EEG data as an MNE Epochs object.

    :return: tuple of mne.EpochsArray and np.ndarray
        A tuple containing the reconstructed EEG epochs, residuals epochs, and matrix A of the estimated coefficients.
        - epochs_recon_fit: mne.EpochsArray
            The reconstructed EEG epochs.
        - epochs_recon_res: mne.EpochsArray
            The residuals epochs.
        - A_full: np.ndarray
            The matrix of estimated coefficients.
    """

    # Create n_channel x n_trials matrix for each timestamp:
    X_full = epochs.get_data() # Retrieves the n_trials x n_channels x n_timestamps data

    # Get all dimensions:
    n_trials, n_channels, n_timestamps = X_full.shape
    n_conditions = S.shape[1]

    X_full_recon = np.empty(X_full.shape)
    X_residuals = np.empty(X_full.shape)
    X_full_recon[:], X_residuals[:] = np.nan, np.nan

    A_full = np.empty((n_conditions, n_channels, n_timestamps), dtype=float)
    A_full[:] = np.nan

    if shrinkage:
        cov = LedoitWolf().fit(S)
        print(cov.covariance_)
        Css_inv = np.linalg.inv(cov.covariance_)
        print(Css_inv)
        print(f'Shrinkage param: {cov.shrinkage_}')
    else:
        pseudo_inv = np.linalg.pinv(S) # Calculate pseudoinverse for S
    for tp in range(n_timestamps):
        X = X_full[:,:,tp] # Get data matrix for current timestamp

        if shrinkage:
            # reg = linear_model.Ridge(alpha=cov.shrinkage_)
            # A = reg.fit(S, X).coef_.T

            # Make X zero mean:
            # X = X - X.mean(axis=0)
            Csx = S.T.dot(X)
            # Csx = calc_cross_cov(S,X) # 5x60

            A = 1/n_trials * Css_inv.dot(Csx)


            # # list of alphas to check: 100 values from 0 to 5 with
            # r_alphas = np.logspace(-6, 6, 13)
            # # initiate the cross validation over alphas
            # ridge_model = RidgeCV(alphas=r_alphas).fit(S,X)
            #
            # print(ridge_model.alpha_)
            # A = ridge_model.coef_.T

            print(f'Timepoint {tp}/{n_timestamps}', end='\r')
        else:
            A = pseudo_inv.dot(X) # Solve inverse problem

        A_full[:,:,tp] = A
        X_hat = S.dot(A) # Get EEG activity explained by conditions
        X_full_recon[:,:,tp] = X_hat # Add to reconstructed EEG
        X_residuals[:,:,tp] = X-X_hat # Add residuals to residuals EEG

    # Create epochs array for the reconstructed EEG, as well as for the residuals:
    epochs_recon_fit = mne.EpochsArray(X_full_recon, info=epochs.info, events=epochs.events, tmin=epochs.tmin, event_id=epochs.event_id, flat=epochs.flat, reject_tmin=epochs.reject_tmin, reject_tmax=epochs.reject_tmax)

    epochs_recon_res = mne.EpochsArray(X_residuals, info=epochs.info, events=epochs.events, tmin=epochs.tmin, event_id=epochs.event_id, flat=epochs.flat, reject_tmin=epochs.reject_tmin, reject_tmax=epochs.reject_tmax)

    return epochs_recon_fit, epochs_recon_res, A_full

def calc_cross_cov(A,B):
    n_trials, n_conditions = A.shape
    n_trials, n_channels = B.shape

    Cab = np.zeros((n_conditions, n_channels))

    # Subtract mean from each trial:
    A_clean = A - A.mean(axis=1).reshape((A.shape[0],1))
    B_clean = B - B.mean(axis=1).reshape((A.shape[0],1))

    for i in range(n_trials):
        Cab += A_clean[i,:].reshape((n_conditions,1))*B_clean[i,:].reshape((n_channels,1)).T
    Cab = 1/n_trials * Cab


    # for j in range(n_conditions):
    #     for k in range(n_channels):
    #         qjk = 0
    #         for i in range(n_trials):
    #             qjk += (A[i,j]-A[i,:].mean()) * (B[i,k]-B[i,:].mean())
    #         qjk = qjk/(n_trials-1)
    #         Cab[j,k] = qjk
    return Cab

def permutation_test(chan, n_cond, n_ts, val_list):
    n_perm = 10000
    n_sbj = len(val_list)
    channel_p_vals = np.zeros((1,n_cond,n_ts))
    sign_list = [-1, 1]
    for ts in range(n_ts):
        for cond in range(n_cond):
            vals = []
            for subj in range(n_sbj):
                vals.append(val_list[subj][cond, chan, ts])

            # Create a random 1, -1 matrix with size len(vals) x n_perm
            signs = np.random.choice(sign_list, size=(len(vals), n_perm))

            # Apply random signs to the vals:
            vals = np.array(np.abs(vals))
            vals = np.reshape(vals, (len(vals), 1))
            vals_rep = np.repeat(vals, n_perm, axis=1)
            vals_to_test = vals_rep * signs

            # Apply one sample t-test:
            _stat, _pval = ttest_1samp(vals_to_test, popmean=0.0, axis=0)

            orig_stat, orig_p = ttest_1samp(vals, popmean=0.0)

            # Sort stats:
            _stat.sort()

            # Check how many values in stats are bigger than the original statistic
            stats_above = _stat > orig_stat

            # Get the number of stats that are bigger than the original statistic:
            ids_above = stats_above.sum()

            # Get proporotion of idcs that are bigger than original statistic:
            channel_p_vals[0, cond, ts] = ids_above / n_perm

    return channel_p_vals


def _perform_permuation_tests(val_list, bootstrap=True, multiproc=True):
    n_cond, n_chan, n_ts = val_list[0].shape
    n_sbj = len(val_list)
    p_vals = np.zeros((n_chan, n_cond, n_ts))

    print(n_chan)

    start = time.time()
    # for chan in range(n_chan):
    p = Pool(processes=n_chan)
    channel_p_vals = p.starmap(permutation_test, zip(range(n_chan), repeat(n_cond), repeat(n_ts), repeat(val_list)))

    # Extract p_vals:
    for ch in range(n_chan):
        # print(channel_ci[0].shape)
        p_vals[ch,:,:] = channel_p_vals[ch][0,:,:]

    print(f'Permutations tests took me: {round((time.time()-start),2)}') #, end='\r')
    return p_vals



def statistical_tests_glm(src, dst, split_id, alignment, shrink=True):
    # List all files with naming convention 'sbj-grand_avg_<sbj>_<alignment>_<condition>
    type = 'regr-coeff'

    # Retrieve all filenames from the source directory that contain regr-coeff and alignement:
    if shrink:
        file_names = [f for f in os.listdir(src) if (type in f) and (alignment in f) and not ('no-shrink' in f)
                      and not ('p-val' in f)]
    else:
        file_names = [f for f in os.listdir(src) if (type in f) and (alignment in f) and ('no-shrink' in f)
                      and not ('p-val' in f)]

    A_matrices = []
    for f in file_names:
        file = src + '/' + f
        # Read all files and combine to grand average:
        A_matrices.append(np.load(file))

    p_vals = _perform_permuation_tests(A_matrices)

    # Store p_vals:
    if shrink:
        store_name = f'{dst}/regr-coeff-p-vals_{alignment}_shrink.npy'
    else:
        store_name = f'{dst}/regr-coeff-p-vals_{alignment}_no-shrink.npy'
    np.save(store_name, p_vals)



def two_sample_tests_glm(src, dst, split_id, alignment, shrink=True):
    # List all files with naming convention 'sbj-grand_avg_<sbj>_<alignment>_<condition>
    type = 'regr-coeff'

    # Retrieve all filenames from the source directory that contain regr-coeff and alignement:
    if shrink:
        file_names = [f for f in os.listdir(src) if (type in f) and (alignment in f) and not ('no-shrink' in f)
                      and not ('p-val' in f)]
    else:
        file_names = [f for f in os.listdir(src) if (type in f) and (alignment in f) and ('no-shrink' in f)
                      and not ('p-val' in f)]

    A_matrices = []
    for f in file_names:
        file = src + '/' + f
        # Read all files and combine to grand average:
        A_matrices.append(np.load(file))

    p_vals = _perform_two_sample_permuation_tests(A_matrices)

    # Store p_vals:
    if shrink:
        store_name = f'{dst}/two-sample-p-vals_{alignment}_shrink.npy'
    else:
        store_name = f'{dst}/two-sample-p-vals_{alignment}_no-shrink.npy'
    np.save(store_name, p_vals)



def _perform_two_sample_permuation_tests(val_list, bootstrap=True, multiproc=True):
    n_cond, n_chan, n_ts = val_list[0].shape
    n_sbj = len(val_list)

    print(n_chan)

    # Overwrite n_cond since we are performing two-sample tests:
    n_cond = 3
    p_vals = np.zeros((n_chan, n_cond, n_ts))

    start = time.time()
    # for chan in range(n_chan):
    p = Pool(processes=n_chan)
    channel_p_vals = p.starmap(permutation_test_two_sample, zip(range(n_chan), repeat(n_cond), repeat(n_ts), repeat(val_list)))

    # Extract p_vals:
    for ch in range(n_chan):
        # print(channel_ci[0].shape)
        p_vals[ch,:,:] = channel_p_vals[ch][0,:,:]

    print(f'Permutations tests took me: {round((time.time()-start),2)}') #, end='\r')
    return p_vals


def permutation_test_two_sample(chan, n_cond, n_ts, val_list):
    n_perm = 10000
    n_sbj = len(val_list)
    channel_p_vals = np.zeros((1,n_cond,n_ts))
    sign_list = [-1, 1]
    for ts in range(n_ts):
        vals_1 = []
        vals_2 = []
        for subj in range(n_sbj):
            vals_1.append(val_list[subj][0, chan, ts])
            vals_2.append(val_list[subj][1, chan, ts])

        # Create a random 1, -1 matrix with size len(vals) x n_perm
        signs = np.random.choice(sign_list, size=(len(vals_1), n_perm))

        # Apply random signs to the vals:
        vals_1 = np.array(vals_1)
        vals_1 = np.reshape(vals_1, (len(vals_1), 1))
        vals_rep_1 = np.repeat(vals_1, n_perm, axis=1)
        vals_to_test_1 = vals_rep_1 * signs

        vals_2 = np.array(vals_2)
        vals_2 = np.reshape(vals_2, (len(vals_2), 1))
        vals_rep_2 = np.repeat(vals_2, n_perm, axis=1)
        vals_to_test_2 = vals_rep_2 * signs

        # Apply one sample t-test:
        _stat, _pval = ttest_ind(vals_to_test_1, vals_to_test_2, axis=0)

        orig_stat, orig_p = ttest_ind(vals_1, vals_2)

        # Sort stats:
        _stat.sort()

        # Check how many values in stats are bigger than the original statistic
        stats_above = _stat > orig_stat

        # Get the number of stats that are bigger than the original statistic:
        ids_above = stats_above.sum()

        # Get proporotion of idcs that are bigger than original statistic:
        channel_p_vals[0, 0, ts] = ids_above / n_perm

        vals_1 = []
        vals_2 = []
        for subj in range(n_sbj):
            vals_1.append(val_list[subj][2, chan, ts])
            vals_2.append(val_list[subj][3, chan, ts])

        # Create a random 1, -1 matrix with size len(vals) x n_perm
        signs = np.random.choice(sign_list, size=(len(vals_1), n_perm))

        # Apply random signs to the vals:
        vals_1 = np.array(vals_1)
        vals_1 = np.reshape(vals_1, (len(vals_1), 1))
        vals_rep_1 = np.repeat(vals_1, n_perm, axis=1)
        vals_to_test_1 = vals_rep_1 * signs

        vals_2 = np.array(vals_2)
        vals_2 = np.reshape(vals_2, (len(vals_2), 1))
        vals_rep_2 = np.repeat(vals_2, n_perm, axis=1)
        vals_to_test_2 = vals_rep_2 * signs

        # Apply one sample t-test:
        _stat, _pval = ttest_ind(vals_to_test_1, vals_to_test_2, axis=0)

        orig_stat, orig_p = ttest_ind(vals_1, vals_2)

        # Sort stats:
        _stat.sort()

        # Check how many values in stats are bigger than the original statistic
        stats_above = _stat > orig_stat

        # Get the number of stats that are bigger than the original statistic:
        ids_above = stats_above.sum()

        # Get proporotion of idcs that are bigger than original statistic:
        channel_p_vals[0, 1, ts] = ids_above / n_perm

        vals_1 = []
        vals_2 = []
        for subj in range(n_sbj):
            vals_1.append(np.abs(val_list[subj][0, chan, ts]))
            vals_1.append(np.abs(val_list[subj][1, chan, ts]))
            vals_2.append(np.abs(val_list[subj][2, chan, ts]))
            vals_2.append(np.abs(val_list[subj][3, chan, ts]))

        # Create a random 1, -1 matrix with size len(vals) x n_perm
        signs = np.random.choice(sign_list, size=(len(vals_1), n_perm))

        # Apply random signs to the vals:
        vals_1 = np.array(vals_1)
        vals_1 = np.reshape(vals_1, (len(vals_1), 1))
        vals_rep_1 = np.repeat(vals_1, n_perm, axis=1)
        vals_to_test_1 = vals_rep_1 * signs

        vals_2 = np.array(vals_2)
        vals_2 = np.reshape(vals_2, (len(vals_2), 1))
        vals_rep_2 = np.repeat(vals_2, n_perm, axis=1)
        vals_to_test_2 = vals_rep_2 * signs

        # Apply one sample t-test:
        _stat, _pval = ttest_ind(vals_to_test_1, vals_to_test_2, axis=0)

        orig_stat, orig_p = ttest_ind(vals_1, vals_2)

        # Sort stats:
        _stat.sort()

        # Check how many values in stats are bigger than the original statistic
        stats_above = _stat > orig_stat

        # Get the number of stats that are bigger than the original statistic:
        ids_above = stats_above.sum()

        # Get proporotion of idcs that are bigger than original statistic:
        channel_p_vals[0, 2, ts] = ids_above / n_perm

    return channel_p_vals



def temporal_processing(src, dst, sbj_list, split=['']):
    pass


def plot_grand_average(src, dst, sbj_list, paradigm, split=[''], plot_topo=False, p_ls=None, times=None):

    # Calculate grand average without conditions:
    if split == ['']:
        avg = [[]]
        title_cond = 'All conditions'

    if split == ['long', 'short']:
        avg = [[],[]]
        title_cond = 'Distance (long v short)'

    if split == ['up', 'down', 'left', 'right']:
        avg = [[],[],[],[]]
        title_cond = 'Direction (up v down v left v right)'

    if split == ['top', 'bottom', 'left', 'right', 'center']:
        avg = [[],[],[],[], []]
        title_cond = 'Positions (top v bottom v left v right v center)'

    # Retrieve all filenames from the source directory:
    file_names = [f for f in os.listdir(src)]

    # Create grand_average np array:
    # grand_avg = np.zeros(())

    diffs_cue_mov = []
    diffs_cue_fin = []
    diffs_start_stop = []

    evokeds_lst = []
    for sbj in sbj_list:
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]

        epochs = mne.read_epochs(file, preload=True)

        diff_cue_release, diff_cue_finished, diff_release_stop = _get_cue_movement_onset_diff(epochs.annotations)
        diffs_cue_mov += diff_cue_release
        diffs_cue_fin += diff_cue_finished
        diffs_start_stop += diff_release_stop

        # Get markers:
        markers = list(epochs.event_id.keys())

        if split == ['']:
            combined_conditions = [m for m in markers]

        if split == ['long', 'short']:
            longs = [m for m in markers if '-l' in m]
            shorts = [m for m in markers if '-s' in m]
            combined_conditions = []
            combined_conditions.append(longs)
            combined_conditions.append(shorts)

        if split == ['up', 'down', 'left', 'right']:
            ups = [m for m in markers if 'BTT-l' in m]
            downs = [m for m in markers if 'TTB-l' in m]
            lefts = [m for m in markers if 'RTL-l' in m]
            rights = [m for m in markers if 'LTR-l' in m]
            combined_conditions = []
            combined_conditions.append(ups)
            combined_conditions.append(downs)
            combined_conditions.append(lefts)
            combined_conditions.append(rights)

        if split == ['top', 'bottom', 'left', 'right', 'center']:
            tops = [m for m in markers if 'TTB-l' in m]
            bots = [m for m in markers if 'BTT-l' in m]
            lefts = [m for m in markers if 'LTR-l' in m]
            rights = [m for m in markers if 'RTL-l' in m]
            centers = [m for m in markers if '-s']
            combined_conditions = []
            combined_conditions.append(tops)
            combined_conditions.append(bots)
            combined_conditions.append(lefts)
            combined_conditions.append(rights)
            combined_conditions.append(centers)

        # Append the average activity of each participant shape = (epochs, channels, times):
        # Averaging all epochs for each timestamp and channel:

        for i, cond in enumerate(split):
            avg[i].append(epochs[combined_conditions[i]].get_data().mean(axis=0))

        evokeds_lst.append(epochs.average())


    # grand_average.plot()
    if plot_topo:
        grand_average = mne.combine_evoked(evokeds_lst, weights='equal')
        fig = grand_average.pick_types(eeg=True).plot_topo(color='r', legend=False)
        fig.savefig(f'{dst}/topoplot_{cond}_{title_cond}.png', dpi=400)
        return

    # mne.viz.plot_compare_evokeds(grand_average, picks='Cz', legend=False)

    grand_avg = []
    uppers_l = []
    lowers_l = []
    for i, cond in enumerate(split):
        grand_avg.append(np.array(avg[i]).mean(axis=0))
        uppers, lowers = _calc_confidence_interval(avg[i], sbj_list)
        uppers_l.append(uppers)
        lowers_l.append(lowers)


    # Get cue, movement onset, movement stop histograms:
    if 'cue_aligned' in src:
        # Make "histogram" of difference between movement onset and cue-alignment:
        bins=np.arange(epochs.tmin, epochs.tmax, 1/epochs.info['sfreq'])
        diffs_cue_mov = np.array(diffs_cue_mov) + 2.0
        diffs_cue_fin = np.array(diffs_cue_fin) + 2.0

        hist_cue_mov = np.histogram(diffs_cue_mov, bins=bins, range=None)
        hist_cue_fin = np.histogram(diffs_cue_fin, bins=bins, range=None)

        l_kernel = 55
        kernel = _gauss(n=l_kernel, b=0.1*epochs.info['sfreq'])

        smoothed_cue_mov = np.convolve(hist_cue_mov[0], kernel, 'same')
        smoothed_cue_fin = np.convolve(hist_cue_fin[0], kernel, 'same')

        smoothed_cue_mov = (smoothed_cue_mov - smoothed_cue_mov.min())/ \
                           (smoothed_cue_mov.max() - smoothed_cue_mov.min())
        smoothed_cue_fin = (smoothed_cue_fin - smoothed_cue_fin.min())/ \
                           (smoothed_cue_fin.max() - smoothed_cue_fin.min())

        # plt.plot(x[1:-1], smoothed_cue_mov)
        # plt.plot(x[1:-1], smoothed_cue_fin)
        # plt.legend(['Release', 'Touch'],
        #            prop={'size': 6}, loc='best')
        # plt.savefig(f'{dst}/difference_between_cue_onset_and_movement_onset.png', dpi=400)

    if 'movement_aligned' in src:
        # Make "histogram" of difference between movement onset and cue-alignment:
        bins=np.arange(epochs.tmin, epochs.tmax, 1/epochs.info['sfreq'])
        diffs_cue_mov = np.array(diffs_cue_mov) * (-1.0)
        diffs_start_stop = np.array(diffs_start_stop)

        hist_cue_mov = np.histogram(diffs_cue_mov, bins=bins, range=None)
        hist_start_stop = np.histogram(diffs_start_stop, bins=bins, range=None)

        l_kernel = 55
        kernel = _gauss(n=l_kernel, b=0.1*epochs.info['sfreq'])

        smoothed_cue_mov = np.convolve(hist_cue_mov[0], kernel, 'same')
        smoothed_start_stop = np.convolve(hist_start_stop[0], kernel, 'same')

        smoothed_cue_mov = (smoothed_cue_mov - smoothed_cue_mov.min())/ \
                           (smoothed_cue_mov.max() - smoothed_cue_mov.min())
        smoothed_start_stop = (smoothed_start_stop - smoothed_start_stop.min())/ \
                              (smoothed_start_stop.max() - smoothed_start_stop.min())


        # plt.plot(x[1:-1], smoothed_cue_mov)
        # plt.plot(x[1:-1], smoothed_start_stop)
        # plt.legend(['Cue', 'Touch'],
        #            prop={'size': 6}, loc='best')
        # plt.savefig(f'{dst}/difference_between_cue_onset_and_movement_onset.png', dpi=400)



    if 'cue_aligned' in src:
        t_zero = 2.0
        title_alignment = 'cue-aligned'

    elif 'movement_aligned' in src:
        t_zero = 0.0
        title_alignment= 'movement-aligned'

    if len(split) != 5:
        print('Before testing')
        p_crit = 0.05
        if len(split) == 2:
            n_chan, n_times = avg[0][0].shape

            p_vals = np.zeros((n_chan, n_times))
            if times is not None:
                n_times = len(times)

            longs = avg[0]
            shorts = avg[1]
            for ch in range(n_chan):
                print(ch)
                for ts in range(n_times):
                    long_to_test = []
                    short_to_test = []
                    for i in range(len(sbj_list)):
                        if times is None:
                            long_to_test.append(longs[i][ch,ts])
                            short_to_test.append(shorts[i][ch,ts])
                        else:
                            sample = int((times[ts] - times[0]) * epochs.info['sfreq'])
                            long_to_test.append(longs[i][ch,sample])
                            short_to_test.append(shorts[i][ch,sample])

                    # res = wilcoxon(long_to_test, short_to_test)
                    # p_vals[ch, ts] = res.pvalue
                    if times is None:
                        p_vals[ch, ts] = perform_permutation_test(long_to_test, short_to_test)
                    else:
                        p_vals[ch, sample] = perform_permutation_test(long_to_test, short_to_test)

            if times is not None:
                return p_vals

        elif len(split) == 4:
            n_chan, n_times = avg[0][0].shape

            # 6 combinations:
            p_ud = np.zeros((n_chan, n_times))
            p_ul = np.zeros((n_chan, n_times))
            p_ur = np.zeros((n_chan, n_times))
            p_dl = np.zeros((n_chan, n_times))
            p_dr = np.zeros((n_chan, n_times))
            p_lr = np.zeros((n_chan, n_times))

            if times is not None:
                n_times = len(times)

            for ch in range(n_chan):
                print(ch)
                for ts in range(n_times):
                    ups_to_test = []
                    downs_to_test = []
                    lefts_to_test = []
                    rights_to_test = []
                    for i in range(len(sbj_list)):
                        if times is None:
                            ups_to_test.append(avg[0][i][ch,ts])
                            downs_to_test.append(avg[1][i][ch,ts])
                            lefts_to_test.append(avg[2][i][ch,ts])
                            rights_to_test.append(avg[3][i][ch,ts])
                        else:
                            sample = int((times[ts] - times[0]) * epochs.info['sfreq'])
                            ups_to_test.append(avg[0][i][ch,sample])
                            downs_to_test.append(avg[1][i][ch,sample])
                            lefts_to_test.append(avg[2][i][ch,sample])
                            rights_to_test.append(avg[3][i][ch,sample])


                    # res = wilcoxon(long_to_test, short_to_test)
                    # p_vals[ch, ts] = res.pvalue
                    if times is None:
                        p_ud[ch, ts] = perform_permutation_test(ups_to_test, downs_to_test)
                        p_ul[ch, ts] = perform_permutation_test(ups_to_test, lefts_to_test)
                        p_ur[ch, ts] = perform_permutation_test(ups_to_test, rights_to_test)
                        p_dl[ch, ts] = perform_permutation_test(downs_to_test, lefts_to_test)
                        p_dr[ch, ts] = perform_permutation_test(downs_to_test, rights_to_test)
                        p_lr[ch, ts] = perform_permutation_test(lefts_to_test, rights_to_test)
                    else:
                        p_ud[ch, sample] = perform_permutation_test(ups_to_test, downs_to_test)
                        p_ul[ch, sample] = perform_permutation_test(ups_to_test, lefts_to_test)
                        p_ur[ch, sample] = perform_permutation_test(ups_to_test, rights_to_test)
                        p_dl[ch, sample] = perform_permutation_test(downs_to_test, lefts_to_test)
                        p_dr[ch, sample] = perform_permutation_test(downs_to_test, rights_to_test)
                        p_lr[ch, sample] = perform_permutation_test(lefts_to_test, rights_to_test)

            if times is not None:
                return p_ud, p_ul, p_ur, p_dl, p_dr, p_lr
    # # Bonferroni correction:
    # p_crit = p_crit / n_times
    # print(p_crit)


    print('Done testing')
    # ch_name = 'Cz'
    # idx = [i for i, name in enumerate(epochs.ch_names) if name == ch_name][0]

    x = np.arange(epochs.tmin, epochs.tmax+1/epochs.info['sfreq'], 1/epochs.info['sfreq'])
    for idx, name in enumerate(epochs.ch_names):
        title = f'{title_cond} {title_alignment} {name}'
        legend_text = []
        if len(split) == 4:
            fig, ax = plt.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [1, 3, 2.5]})
        else:
            fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3]})
        for i, cond in enumerate(split):
            ax[1].plot(x, grand_avg[i][idx, :]*1e6, linewidth=0.5)
            ax[1].fill_between(x, lowers_l[i][idx, :]*1e6, uppers_l[i][idx, :]*1e6, alpha=0.1)
            if len(split) == 1:
                legend_text.append('Grand average')
            else:
                legend_text.append(f'{cond}')
            legend_text.append('95%-CI')

        ax[1].plot([t_zero, t_zero], [lowers_l[i][idx, :].min()*1e6, uppers_l[i][idx, :].max()*1e6], color='black')
        legend_text.append( 'Cue presentation')
        # Plot line between conditions if pval is  smaller than p_crit:
        if len(split) == 2:
            for ts, p in enumerate(p_vals[idx, :]):
                if p < p_crit:
                    ax[1].plot([x[ts], x[ts]], [grand_avg[0][idx, ts]*1e6, grand_avg[1][idx, ts]*1e6], color='lightgreen')

        if len(split) == 4:
            if p_ls is not None:
                for ts, p in enumerate(p_ls[idx, :]):
                    if p < p_crit:
                        ax[2].plot([x[ts], x[ts]], [6.75, 7.25], color='blue')
            for ts, p in enumerate(p_ud[idx, :]):
                if p < p_crit:
                    ax[2].plot([x[ts], x[ts]], [5.75, 6.25], color='brown')
            for ts, p in enumerate(p_ul[idx, :]):
                if p < p_crit:
                    ax[2].plot([x[ts], x[ts]], [4.75, 5.25], color='darkgrey')
            for ts, p in enumerate(p_ur[idx, :]):
                if p < p_crit:
                    ax[2].plot([x[ts], x[ts]], [3.75, 4.25], color='yellow')
            for ts, p in enumerate(p_dl[idx, :]):
                if p < p_crit:
                    ax[2].plot([x[ts], x[ts]], [2.75, 3.25], color='lime')
            for ts, p in enumerate(p_dr[idx, :]):
                if p < p_crit:
                    ax[2].plot([x[ts], x[ts]], [1.75, 2.25], color='cyan')
            for ts, p in enumerate(p_lr[idx, :]):
                if p < p_crit:
                    ax[2].plot([x[ts], x[ts]], [0.75, 1.25], color='magenta')

            ax[2].set_ylim([0,8])
            ax[2].set_yticks([1,2,3,4,5,6,7])
            ax[2].set_xlim([x[0], x[-1]])

            lbls = ['Left vs. Right', 'Down vs. Right', 'Down vs. Left', 'Up vs. Right', 'Up vs. Left', 'Up vs. Down', 'Long vs. Short']
            ax[2].set_yticklabels(lbls)
            # labels = [item.get_text() for item in ax[2].get_yticklabels()]

            # ax.set_xticklabels(labels)



        # plt.plot(x,grand_avg_short[idx,:]*1e6)
        # plt.fill_between(x, lowers_short[idx,:]*1e6, uppers_short[idx,:]*1e6, alpha=0.1)

        ax[1].legend(legend_text, loc='center left', prop={'size': 6}, bbox_to_anchor=(1, 0.5))
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Voltage (uV)')
        ax[1].set_xlim([x[0], x[-1]])
        fig.suptitle(title)

        if 'cue_aligned' in src:
            ax[0].plot(x[1:-1], smoothed_cue_mov)
            ax[0].plot(x[1:-1], smoothed_cue_fin)
            ax[0].legend(['Release', 'Touch'],
                         prop={'size': 6}, loc='center left', bbox_to_anchor=(1, 0.5))

        if 'movement_aligned' in src:
            ax[0].plot(x[1:-1], smoothed_cue_mov)
            ax[0].plot(x[1:-1], smoothed_start_stop)
            ax[0].legend(['Cue', 'Touch'], prop={'size': 6}, loc='center left', bbox_to_anchor=(1, 0.5))
            # pass

        ax[0].set_xlim([x[0], x[-1]])
        plt.tight_layout()
        plt.savefig(f'{dst}/grand_average_{name}_{title_alignment}_{title_cond}.png', dpi=400)
        plt.close('all')
        # plt.show()




    # plt.plot(range(grand_avg.shape[1]), grand_avg[13,:])
    # plt.show()

    # _calc_grand_average([avg])
    if len(split) == 2:
        return p_vals
    else:
        return avg