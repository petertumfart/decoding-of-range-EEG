import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def glm_significant_topo(src, dst, sbj_list, alignment, p_crit=.05, shrink=True, one_sample=True):
    # Load p_vals of glm:
    if shrink:
        if one_sample:
            p_vals = np.load(f'{src}/regr-coeff-p-vals_{alignment}_shrink.npy')
        else:
            p_vals = np.load(f'{src}/two-sample-p-vals_{alignment}_shrink.npy')
    else:
        if one_sample:
            p_vals = np.load(f'{src}/regr-coeff-p-vals_{alignment}_no-shrink.npy')
        else:
            p_vals = np.load(f'{src}/two-sample-p-vals_{alignment}_no-shrink.npy')

    # Get binary mask for p_vals that are smaller than p_crit
    p_bin = p_vals < p_crit

    # Load sample epoch:
    epochs = mne.read_epochs(f'{src}/sample-epoch_{alignment}_epo.fif', preload=True)

    # Load grand average:
    grand_avg = np.load(f'{src}/grand-avg_{alignment}_All-conditions.npy')

    n_chan, n_cond, n_times = p_vals.shape

    for cond in range(n_cond):
        temp_avg = np.copy(grand_avg)
        # Set grand average to zero where p_bin is False:
        temp_avg[np.where(p_bin[:,cond,:]==False)] = 0.0

        # Add grand average into mne structure:
        grand_avg_extended = temp_avg.reshape((1, n_chan, n_times))
        grand_avg_epochs = mne.EpochsArray(grand_avg_extended, epochs.info, tmin=epochs.tmin)

        if alignment == 'cue-aligned':
            times = np.arange(2.0, 3.0, 0.1)
        elif alignment == 'movement-aligned':
            times = np.arange(-.5, .5, .1)

        fig = grand_avg_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10,
                                                      nrows='auto', image_interp='linear')#, scalings=dict(eeg=1e-6))#, units='a.u.', vlim=(0, 1))
        if shrink:
            if one_sample:
                fig.savefig(f'{dst}/glm-significant-topo_{alignment}_shrink_{cond}_one-sample.png', dpi=400)
            else:
                fig.savefig(f'{dst}/glm-significant-topo_{alignment}_shrink_{cond}_two-sample.png', dpi=400)
        else:
            if one_sample:
                fig.savefig(f'{dst}/glm-significant-topo_{alignment}_no-shrink_{cond}_one-sample.png', dpi=400)
            else:
                fig.savefig(f'{dst}/glm-significant-topo_{alignment}_no-shrink_{cond}_two-sample.png', dpi=400)

        if cond == 0:
            grand_1 = temp_avg.copy()
        if cond == 1:
            grand_2 = temp_avg.copy()

    # temp_avg = (grand_1 - grand_2)
    # # Add grand average into mne structure:
    # grand_avg_extended = temp_avg.reshape((1, n_chan, n_times))
    # grand_avg_epochs = mne.EpochsArray(grand_avg_extended, epochs.info, tmin=epochs.tmin)
    #
    # cond = 6
    # fig = grand_avg_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10,
    #                                               nrows='auto', image_interp='linear')#, scalings=dict(eeg=1e-6))#, units='a.u.', vlim=(0, 1))
    # if shrink:
    #     fig.savefig(f'{dst}/glm-significant-topo_{alignment}_shrink_{cond}.png', dpi=400)
    # else:
    #     fig.savefig(f'{dst}/glm-significant-topo_{alignment}_no-shrink_{cond}.png', dpi=400)

def condition_topos(src, dst, sbj_list, alignment, shrink=True):
    # Retrieve all filenames from the source directory:
    if shrink:
        file_names = [f for f in os.listdir(src) if ('regr-coeff' in f) and (alignment in f) and not ('no-shrink' in f)
                      and not ('p_vals' in f)]
        param_file_names = [f for f in os.listdir(src) if ('param-matrix' in f) and (alignment in f)
                            and not ('no-shrink' in f)]
    else:
        file_names = [f for f in os.listdir(src) if ('regr-coeff' in f) and (alignment in f) and ('no-shrink' in f)
                      and not ('p_vals' in f)]
        param_file_names = [f for f in os.listdir(src) if ('param-matrix' in f) and (alignment in f)
                            and ('no-shrink' in f)]

    # Load sample epoch:
    epochs = mne.read_epochs(f'{src}/sample-epoch_{alignment}_epo.fif', preload=True)

    sbj_condition_eeg = []
    for sbj in sbj_list:
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]
        param_file = src + '/' + [f for f in param_file_names if (sbj in f)][0]
        print(param_file)
        # Load regression coefficients:
        A = np.load(file)
        S = np.load(param_file)

        n_cond, n_chan, n_ts = A.shape
        n_epochs, n_cond = S.shape

        condition_eeg_list = []
        for cond in range(n_cond):
            condition_eeg = np.zeros((n_chan, n_ts))
            for ts in range(n_ts):
                condition_eeg[:,ts] = (A[cond,:,ts].reshape((n_chan, 1))*S[:,cond].reshape((1, n_epochs))).mean(axis=1)
            condition_eeg_list.append(condition_eeg)

        sbj_condition_eeg.append(condition_eeg_list)

    # Calc average and std:
    avg_cond = []
    for cond in range(n_cond):
        grand_average = np.zeros((n_chan, n_ts))
        for sbj in range(len(sbj_list)):
            grand_average += sbj_condition_eeg[sbj][cond]

        grand_average = grand_average / len(sbj_list)
        avg_cond.append(grand_average)


    avg_std = []
    for cond in range(n_cond):
        grand_standard_dev = np.zeros((n_chan, n_ts))
        for sbj in range(len(sbj_list)):
            grand_standard_dev += (sbj_condition_eeg[sbj][cond] - avg_cond[cond])**2

        grand_standard_dev = grand_standard_dev / len(sbj_list)
        grand_standard_dev = grand_standard_dev ** 0.5
        avg_std.append(grand_standard_dev)

    for cond in range(n_cond):
        # Add them into the mne structure:
        grand_avg_extended = avg_cond[cond].reshape((1, n_chan, n_ts))
        grand_avg_epochs = mne.EpochsArray(grand_avg_extended, epochs.info, tmin=epochs.tmin)

        grand_standard_dev_extended = avg_std[cond].reshape((1, n_chan, n_ts))
        grand_standard_dev_epochs = mne.EpochsArray(grand_standard_dev_extended, epochs.info, tmin=epochs.tmin)

        if alignment == 'cue-aligned':
            times = np.arange(2.0, 3.0, 0.1)
        elif alignment == 'movement-aligned':
            times = np.arange(-.5, .5, .1)

        # Topoplot these results:
        fig = grand_avg_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10, nrows='auto')#, scalings=dict(eeg=1e-6))#, units='a.u.', vlim=(0, 1))
        fig = grand_standard_dev_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10, nrows='auto')

        # fig.savefig(f'{dst}/glm-significant-topo_{alignment}_{cond}.png', dpi=400)

    # Combine long-short
    # Add them into the mne structure:
    avg_range = (avg_cond[0] + avg_cond[1]) / 2
    avg_range_extended = avg_range.reshape((1, n_chan, n_ts))
    avg_range_epochs = mne.EpochsArray(avg_range_extended, epochs.info, tmin=epochs.tmin)

    avg_std_range = (avg_std[0] + avg_std[1]) / 2
    avg_std_range_extended = avg_std_range.reshape((1, n_chan, n_ts))
    avg_std_range_epochs = mne.EpochsArray(avg_std_range_extended, epochs.info, tmin=epochs.tmin)

    # Topoplot these results:
    fig = avg_range_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10, nrows='auto')#, scalings=dict(eeg=1e-6))#, units='a.u.', vlim=(0, 1))
    fig = avg_std_range_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10, nrows='auto')


    # Combine horz-vert
    # Add them into the mne structure:
    avg_dir = (avg_cond[2] + avg_cond[3]) / 2
    avg_dir_extended = avg_dir.reshape((1, n_chan, n_ts))
    avg_dir_epochs = mne.EpochsArray(avg_dir_extended, epochs.info, tmin=epochs.tmin)

    avg_std_dir = (avg_std[2] + avg_std[3]) / 2
    avg_std_dir_extended = avg_std_dir.reshape((1, n_chan, n_ts))
    avg_std_dir_epochs = mne.EpochsArray(avg_std_dir_extended, epochs.info, tmin=epochs.tmin)

    # Topoplot these results:
    fig = avg_dir_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10, nrows='auto')#, scalings=dict(eeg=1e-6))#, units='a.u.', vlim=(0, 1))
    fig = avg_std_dir_epochs.average().plot_topomap(times, ch_type='eeg', ncols=10, nrows='auto')














