import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timezone
from scipy.stats import t
import scipy.io
from scipy.stats import wilcoxon, ttest_ind
import random
import time


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


def _calc_confidence_interval(avg, bootstrap=True):
    n_bootstrap = 1000
    n_chan, n_ts = avg[0].shape
    n_sbj = len(avg)
    uppers = np.zeros((n_chan, n_ts))
    lowers = np.zeros((n_chan, n_ts))

    confidence = .95
    for chan in range(n_chan):
        start = time.time()
        for ts in range(n_ts):
            vals = []
            for subj in range(n_sbj):
                vals.append(avg[subj][chan, ts])

            if bootstrap:
                # Bootstrapping for confidence interval:
                values = [np.random.choice(vals, size=len(vals), replace=True).mean() for i in range(n_bootstrap)]
                lowers[chan,ts], uppers[chan,ts] = np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)])
            else:
                m = np.array(vals).mean()
                s = np.array(vals).std()
                dof = len(vals)-1

                t_crit = np.abs(t.ppf((1-confidence)/2,dof))

                lowers[chan, ts], uppers[chan, ts] = (m-s*t_crit/np.sqrt(len(vals)), m+s*t_crit/np.sqrt(len(vals)))
        print(f'One channel took me: {round((time.time()-start),2)} chan: {chan:02}', end='\r')
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

        cue_times, release_times, touch_times = _get_cue_movement_onset_diff(epochs.annotations)

        # Store the subject grand averages:
        store_name = f'{dst}/timings-cue_{sbj}_{alignment}.npy'
        np.save(store_name, cue_times)
        store_name = f'{dst}/timings-release_{sbj}_{alignment}.npy'
        np.save(store_name, cue_times)
        store_name = f'{dst}/timings-touch_{sbj}_{alignment}.npy'
        np.save(store_name, cue_times)


def _get_cue_movement_onset_diff(annot):
    # Get difference between cue onset and movement onset (*i*1):
    trial_type_markers = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
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