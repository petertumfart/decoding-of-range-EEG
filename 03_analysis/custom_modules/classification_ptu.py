import os
import mne
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.covariance import LedoitWolf
from scipy import stats
from datetime import datetime, timezone


def classify(src, dst, sbj, condition, n_timepoints=1, loo=True):
    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f)][0]

    file = src + '/' + f_name
    epochs = mne.read_epochs(file, preload=True)

    df_scores = _create_scores_df()

    if 'cue' in src:
        epoch_type = 'cue-aligned'
    elif 'movement' in src:
        epoch_type = 'movement_aligned'

    markers = list(epochs.event_id.keys())
    if condition == 'distance':
        longs = [m for m in markers if '-l' in m]
        shorts = [m for m in markers if '-s' in m]
        epochs_long = epochs[longs]
        epochs_short = epochs[shorts]

        # Equalize epochs:
        n_long = len(epochs_long)
        n_short = len(epochs_short)

        min_len = min(n_long, n_short)

        ids_long = sorted(random.sample(range(n_long), k=min_len))
        ids_short = sorted(random.sample(range(n_short), k=min_len))

        epochs_long = epochs_long[ids_long]
        epochs_short = epochs_short[ids_short]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_long.get_data(), epochs_short.get_data()])
        y = np.concatenate([np.zeros(len(epochs_long)), np.ones(len(epochs_short))])

    elif condition == 'direction':
        # Get condition:
        ups = [m for m in markers if 'BT' in m]
        downs = [m for m in markers if 'TT' in m]
        lefts = [m for m in markers if 'RT' in m]
        rights = [m for m in markers if 'LT' in m]
        epochs_up = epochs[ups]
        epochs_down = epochs[downs]
        epochs_right = epochs[rights]
        epochs_left = epochs[lefts]

        # Equalize epochs:
        n_up = len(epochs_up)
        n_down = len(epochs_down)
        n_left = len(epochs_left)
        n_right = len(epochs_right)

        min_len = min(n_up, n_down, n_left, n_right)

        ids_top = sorted(random.sample(range(n_up), k=min_len))
        ids_bot = sorted(random.sample(range(n_down), k=min_len))
        ids_left = sorted(random.sample(range(n_left), k=min_len))
        ids_right = sorted(random.sample(range(n_right), k=min_len))

        epochs_up = epochs_up[ids_top]
        epochs_down = epochs_down[ids_bot]
        epochs_left = epochs_left[ids_left]
        epochs_right = epochs_right[ids_right]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_up.get_data(), epochs_down.get_data(), epochs_right.get_data(), epochs_left.get_data()])
        y = np.concatenate([np.zeros(len(epochs_up)), np.ones(len(epochs_down)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left))])

    elif condition == 'direction_short':
        ups = [m for m in markers if 'BTT-s' in m]
        downs = [m for m in markers if 'TTB-s' in m]
        lefts = [m for m in markers if 'RTL-s' in m]
        rights = [m for m in markers if 'LTR-s' in m]
        epochs_up = epochs[ups]
        epochs_down = epochs[downs]
        epochs_right = epochs[rights]
        epochs_left = epochs[lefts]

        # Equalize epochs:
        n_up = len(epochs_up)
        n_down = len(epochs_down)
        n_left = len(epochs_left)
        n_right = len(epochs_right)

        min_len = min(n_up, n_down, n_left, n_right)

        ids_top = sorted(random.sample(range(n_up), k=min_len))
        ids_bot = sorted(random.sample(range(n_down), k=min_len))
        ids_left = sorted(random.sample(range(n_left), k=min_len))
        ids_right = sorted(random.sample(range(n_right), k=min_len))

        epochs_up = epochs_up[ids_top]
        epochs_down = epochs_down[ids_bot]
        epochs_left = epochs_left[ids_left]
        epochs_right = epochs_right[ids_right]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_up.get_data(), epochs_down.get_data(), epochs_right.get_data(), epochs_left.get_data()])
        y = np.concatenate([np.zeros(len(epochs_up)), np.ones(len(epochs_down)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left))])

    elif condition == 'direction_long':
        ups = [m for m in markers if 'BTT-l' in m]
        downs = [m for m in markers if 'TTB-l' in m]
        lefts = [m for m in markers if 'RTL-l' in m]
        rights = [m for m in markers if 'LTR-l' in m]
        epochs_up = epochs[ups]
        epochs_down = epochs[downs]
        epochs_right = epochs[rights]
        epochs_left = epochs[lefts]

        # Equalize epochs:
        n_up = len(epochs_up)
        n_down = len(epochs_down)
        n_left = len(epochs_left)
        n_right = len(epochs_right)

        min_len = min(n_up, n_down, n_left, n_right)

        ids_top = sorted(random.sample(range(n_up), k=min_len))
        ids_bot = sorted(random.sample(range(n_down), k=min_len))
        ids_left = sorted(random.sample(range(n_left), k=min_len))
        ids_right = sorted(random.sample(range(n_right), k=min_len))

        epochs_up = epochs_up[ids_top]
        epochs_down = epochs_down[ids_bot]
        epochs_left = epochs_left[ids_left]
        epochs_right = epochs_right[ids_right]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_up.get_data(), epochs_down.get_data(), epochs_right.get_data(), epochs_left.get_data()])
        y = np.concatenate([np.zeros(len(epochs_up)), np.ones(len(epochs_down)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left))])

    elif condition == 'position':
        tops = [m for m in markers if 'BTT-l' in m]
        bottoms = [m for m in markers if 'TTB-l' in m]
        lefts = [m for m in markers if 'RTL-l' in m]
        rights = [m for m in markers if 'LTR-l' in m]
        centers = [m for m in markers if '-s' in m]
        epochs_top= epochs[tops]
        epochs_bottom = epochs[bottoms]
        epochs_right = epochs[rights]
        epochs_left = epochs[lefts]
        epochs_center = epochs[centers]

        # Equalize epochs:
        n_top = len(epochs_top)
        n_bot = len(epochs_bottom)
        n_left = len(epochs_left)
        n_right = len(epochs_right)
        n_center = len(epochs_center)

        min_len = min(n_top, n_bot, n_left, n_right, n_center)

        ids_top = sorted(random.sample(range(n_top), k=min_len))
        ids_bot = sorted(random.sample(range(n_bot), k=min_len))
        ids_left = sorted(random.sample(range(n_left), k=min_len))
        ids_right = sorted(random.sample(range(n_right), k=min_len))
        ids_center = sorted(random.sample(range(n_center), k=min_len))

        epochs_top = epochs_top[ids_top]
        epochs_bottom = epochs_bottom[ids_bot]
        epochs_left = epochs_left[ids_left]
        epochs_right = epochs_right[ids_right]
        epochs_center = epochs_center[ids_center]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_top.get_data(), epochs_bottom.get_data(), epochs_right.get_data(), epochs_left.get_data(), epochs_center.get_data()])
        y = np.concatenate([np.zeros(len(epochs_top)), np.ones(len(epochs_bottom)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left)), 4*np.ones(len(epochs_center))])

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    n_len = X.shape[2] - n_timepoints + 1
    for tp in range(n_timepoints, X.shape[2]+1):
        x = X[:,:,tp-n_timepoints:tp]
        x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))

        if loo:
            scores = cross_val_score(clf, x, y, cv=LeaveOneOut(), n_jobs=-1)
        else:
            scores = cross_val_score(clf, x, y, cv=5, n_jobs=-1)

        # Add row to the dataframe:
        row_to_add = {'Timepoint': (tp-1)/10 + epochs.tmin, 'Accuracy': scores.mean(), 'Subject': sbj,
                      'N_timepoints': n_timepoints, 'Type': epoch_type, 'Init_marker': [markers],
                      't_min': epochs.tmin, 't_max': epochs.tmax, 'epoch_info': [epochs.info],
                      'Date':datetime.now().strftime('%Y-%m-%d'), 'Time': datetime.now().strftime('%H:%M:%S'),
                      'Condition': condition}
        df_scores = pd.concat([df_scores, pd.DataFrame(row_to_add)], ignore_index=True)

        if tp != n_len:
            print(f'Measuring timestamp {tp}/{n_len}, shape: {x.shape}', end='\r')
        else:
            print(f'Measuring timestamp {tp}/{n_len}')

    print()

    # Add mean of scores as subject: Mean:
    # row_to_add = {'Timepoint': (np.arange(n_timepoints-1, X.shape[2])/10 + epochs.tmin).tolist(),
    #               'Accuracy': df_scores.groupby('Timepoint')['Accuracy'].mean().to_list(),
    #               'Subject': ['Mean']*n_len, 'N_timepoints': [n_timepoints]*n_len, 'Type': [epoch_type]*n_len,
    #               'Init_marker': [markers]*n_len, 't_min': [epochs.tmin]*n_len, 't_max': [epochs.tmax]*n_len,
    #               'epoch_info': [[epochs.info]]*n_len, 'Date':[datetime.now().strftime('%Y-%m-%d')]*n_len,
    #               'Time': [datetime.now().strftime('%H:%M:%S')]*n_len, 'Condition': [condition]*n_len]}

    # df_scores = pd.concat([df_scores, pd.DataFrame(row_to_add)], ignore_index=True)

    # Store dataframe to full classification dataframe:
    if loo:
        _store_scores_df(df_scores, csv_name='dataframes/classification/classification_df.csv')
    else:
        _store_scores_df(df_scores, csv_name='dataframes/classification/classification_df_5_fold.csv')

def get_confusion_matrix(src, dst, sbj_list, ts_of_interest, n_timepoints=1, condition='distance'):
    if 'cue' in src:
        epoch_type = 'cue_aligned'
    elif 'movement' in src:
        epoch_type = 'movement_aligned'

    # Retrieve all filenames from the source directory:
    file_names = [f for f in os.listdir(src)]

    conf_mat = []

    for j, sbj in enumerate(sbj_list):
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]

        epochs = mne.read_epochs(file, preload=True)

        if condition == 'distance':
            vmax = 0.5
            # Only made for one condition for now:
            markers = list(epochs.event_id.keys())
            longs = [m for m in markers if '-l' in m]
            shorts = [m for m in markers if '-s' in m]
            epochs_long = epochs[longs]
            epochs_short = epochs[shorts]
            labels = ['Long', 'Short']

            # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
            X = np.concatenate([epochs_long.get_data(), epochs_short.get_data()])
            y = np.concatenate([np.zeros(len(epochs_long)), np.ones(len(epochs_short))])

        elif condition == 'direction':
            vmax = 0.25
            # Only made for one condition for now:
            markers = list(epochs.event_id.keys())
            ups = [m for m in markers if 'BTT-s' in m]
            downs = [m for m in markers if 'TTB-s' in m]
            lefts = [m for m in markers if 'RTL-s' in m]
            rights = [m for m in markers if 'LTR-s' in m]
            epochs_up = epochs[ups]
            epochs_down = epochs[downs]
            epochs_right = epochs[rights]
            epochs_left = epochs[lefts]
            labels = ['Up', 'Down', 'Right', 'Left']

            # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
            X = np.concatenate([epochs_up.get_data(), epochs_down.get_data(), epochs_right.get_data(), epochs_left.get_data()])
            y = np.concatenate([np.zeros(len(epochs_up)), np.ones(len(epochs_down)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left))])

        clf = LinearDiscriminantAnalysis(solver='svd')#, shrinkage='auto')

        # Get x data for peak sample:
        peak_sample = int((ts_of_interest[j] - epochs.tmin)*epochs.info['sfreq'])
        print(peak_sample)

        x = X[:,:,peak_sample-n_timepoints:peak_sample+1]
        x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))

        y_pred = cross_val_predict(clf, x, y, cv=LeaveOneOut(), n_jobs=-1)
        # print(y)
        # print(y_pred-y)
        print(y.shape)
        print(y_pred.shape)
        conf_mat.append(confusion_matrix(y, y_pred, normalize='all'))#/len(epochs))


    # Calculate mean conf_mat:
    mean_conf_mat = np.zeros((conf_mat[0].shape))
    for conf in conf_mat:
        mean_conf_mat += conf

    mean_conf_mat = mean_conf_mat/len(sbj_list)

    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(mean_conf_mat, cmap='Greens',  vmin=0, vmax=vmax)
    for (i, j), z in np.ndenumerate(mean_conf_mat):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)

    # plt.colorbar()
    plt.savefig(f'{dst}/confusion_matrix_{epoch_type}_at_ts_{ts_of_interest}_for_{n_timepoints}_window_{condition}_short_cue.png', dpi=400)
    plt.clf()
    plt.close()



def _create_scores_df():
    # Create dataframe for storing all the classification data + information:
    cols = ['Timepoint',   # Timepoint of classification accuracy
            'Accuracy',    # Classification accuracy
            'Subject',     # Subject ID
            'N_timepoints',     # Number of timepoints used for classification
            'Type',        # Cue-aligned or Movement onset-aligned
            'Init_marker', # Marker(s) used for epoching
            't_min',
            't_max',
            'epoch_info',
            'Date',
            'Time',
            'Condition']

    df = pd.DataFrame(columns=cols)

    return df


def _store_scores_df(df_to_append, csv_name='classification_df.csv'):
    # Check if dataframe exists and if not, create it:
    if not os.path.exists(csv_name):
        df_to_append.to_csv(csv_name)
        return df_to_append
    else:
        df = pd.read_csv(csv_name, index_col=0)
        df = pd.concat([df, df_to_append], ignore_index=True)
        df.to_csv(csv_name)
        return df


def glm(src, dst, sbj):
    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f)][0]

    if 'cue_aligned' in src:
        title_alignment = 'cue_aligned'
    if 'movement_aligned' in src:
        title_alignment = 'movement_aligned'

    file = src + '/' + f_name
    epochs = mne.read_epochs(file, preload=True)

    S = _create_parameter_matrix(epochs, z_scoring=True)

    _,_,A = _fit_glm(S, epochs, shrinkage=True)

    # Save regression coefficients:
    store_name = f'{dst}/{sbj}_regr_coeff_{title_alignment}_shrink.npy'
    np.save(store_name, A)

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
        s_short = (s_short - s_short.mean())/s_short.std()
        s_long = (s_long - s_long.mean())/s_long.std()
        s_vert = (s_vert - s_vert.mean())/s_vert.std()
        s_horz = (s_horz - s_horz.mean())/s_horz.std()

    # Return S matrix:
    return np.concatenate((s_short, s_long, s_vert, s_horz, s_intercept),axis=0)

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
    n_conditions = S.shape[0]

    X_full_recon = np.empty(X_full.shape)
    X_residuals = np.empty(X_full.shape)
    X_full_recon[:], X_residuals[:] = np.nan, np.nan

    A_full = np.empty((n_channels, n_conditions, n_timestamps), dtype=float)
    A_full[:] = np.nan

    if shrinkage:
        cov = LedoitWolf().fit(S.T)
        Css_inv = np.linalg.inv(cov.covariance_)
        print(f'Shrinkage param: {cov.shrinkage_}')
    else:
        pseudo_inv = np.linalg.pinv(S) # Calculate pseudoinverse for S
    for tp in range(len(epochs.times)):
        X = X_full[:,:,tp].T # Get data matrix for current timestamp

        if shrinkage:
            Cxs = X.dot(S.T)/np.trace(X.dot(S.T))
            A = Cxs.dot(Css_inv)
        else:
            A = X.dot(pseudo_inv) # Solve inverse problem

        A_full[:,:,tp] = A
        X_hat = A.dot(S) # Get EEG activity explained by conditions
        X_full_recon[:,:,tp] = X_hat.T # Add to reconstructed EEG
        X_residuals[:,:,tp] = X.T-X_hat.T # Add residuals to residuals EEG

    # Create epochs array for the reconstructed EEG, as well as for the residuals:
    epochs_recon_fit = mne.EpochsArray(X_full_recon, info=epochs.info, events=epochs.events, tmin=epochs.tmin, event_id=epochs.event_id, flat=epochs.flat, reject_tmin=epochs.reject_tmin, reject_tmax=epochs.reject_tmax)

    epochs_recon_res = mne.EpochsArray(X_residuals, info=epochs.info, events=epochs.events, tmin=epochs.tmin, event_id=epochs.event_id, flat=epochs.flat, reject_tmin=epochs.reject_tmin, reject_tmax=epochs.reject_tmax)

    return epochs_recon_fit, epochs_recon_res, A_full


def coefficient_testing(src, dst, sbj_list, two_sample=True):
    # Retrieve all filenames from the source directory:
    file_names = [f for f in os.listdir(src)]

    A = []
    for sbj in sbj_list:
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]

        A.append(np.load(file))

    # Calculate mean for each channel and for each timestamp:
    n_chan, n_coeff, n_ts = A[0].shape
    means = np.empty((n_chan, n_coeff, n_ts))
    uppers = np.empty((n_chan, n_coeff, n_ts))
    lowers = np.empty((n_chan, n_coeff, n_ts))
    means[:], uppers[:], lowers[:] = np.nan, np.nan, np.nan

    stat = np.empty((n_chan, n_ts))
    p_val = np.empty((n_chan, n_ts))
    stat[:], p_val[:] = np.nan, np.nan

    confidence = 0.95
    for ts in range(n_ts):
        for chan in range(n_chan):
            print(f'{ts+1}/{n_ts}, {chan+1}/{n_chan}', end='\r')
            for coeff in range(n_coeff):
                a = [A[i][chan, coeff, ts] for i in range(len(sbj_list))] # Get list of all subject for a specific channel, coefficient and timestamp

                # Calculate mean:
                # Bootstrapping for confidence interval:
                values = [np.random.choice(a, size=len(a),replace=True).mean() for i in range(500)]
                means[chan,coeff,ts] = np.array(values).mean()
                lowers[chan,coeff,ts], uppers[chan,coeff,ts] = np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)])

            rvs1 = [A[i][chan, 1, ts] for i in range(len(sbj_list))]
            rvs2 = [A[i][chan, 2, ts] for i in range(len(sbj_list))]

            if two_sample:
                # Calculate t-test for coeff[1] vs coeff[2] (distance vs. direction):
                stat[chan,ts], p_val[chan, ts] = stats.ttest_ind(rvs1, rvs2)
            else:
                stat[chan,ts], p_val[chan, ts] = stats.ttest_1samp(rvs2, popmean=0)


    # Save bootstrapping matrices:
    np.save(f'{dst}/regr_coeff_global_means.npy', means)    # .npy extension is added if not given
    np.save(f'{dst}/regr_coeff_global_lowers.npy', lowers)
    np.save(f'{dst}/regr_coeff_global_uppers.npy', uppers)

    # Save t-test matrices:
    np.save(f'{dst}/regr_coeff_stats.npy', stat)
    np.save(f'{dst}/regr_coeff_pval.npy', p_val)


def plot_heatmap_of_regr_coeff(src, dst, p_crit=.05):
    # Load bootstrapping matrices:
    means = np.load(f'{src}/regr_coeff_global_means.npy')    # .npy extension is added if not given
    lowers = np.load(f'{src}/regr_coeff_global_lowers.npy')
    uppers = np.load(f'{src}/regr_coeff_global_uppers.npy')

    # Save t-test matrices:
    stat = np.load(f'{src}/regr_coeff_stats.npy')
    p_val = np.load(f'{src}/regr_coeff_pval.npy')

    ch_names = np.load(f'{src}/ch_names.npy')

    n_chan, n_coeff, n_ts = means.shape

    # Generate binary matrix for a distinct p-val:
    bin_p_val = np.empty((n_chan,n_ts))
    bin_p_val[:] = False
    bin_p_val[np.where(p_val < p_crit)] = True

    if 'cue_aligned' in src:
        title_alignment = 'cue_aligned'
        x = np.arange(0.0,n_ts)/10
    if 'movement_aligned' in src:
        title_alignment = 'movement_aligned'
        x = np.arange(-20,n_ts-20)/10

    fig, ax = plt.subplots()
    ax.imshow(bin_p_val) #, cmap='Blues')
    # ax.colorbar()
    ax.yaxis.set_ticks([i for i in range(len(ch_names))])
    ax.set_yticklabels(ch_names)
    ax.xaxis.set_ticks([ts for ts in range(n_ts) if ts % 10 == 0])
    ax.set_xticklabels(x[ts] for ts in range(n_ts) if ts % 10 == 0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels')
    fig.suptitle(f'Difference between distance and direction\nencoded regression coefficients. p-value={p_crit}')



    plt.savefig(f'{dst}/regression_coefficient_heatmap_{title_alignment}.png', dpi=400)
    plt.show()
    plt.close('all')



def classify_single_channel(src, dst, sbj, condition, n_timepoints=1):
    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f)][0]

    file = src + '/' + f_name
    epochs = mne.read_epochs(file, preload=True)

    df_scores = _create_scores_df()

    if 'cue' in src:
        epoch_type = 'cue-aligned'
    elif 'movement' in src:
        epoch_type = 'movement_aligned'

    markers = list(epochs.event_id.keys())
    if condition == 'distance':
        longs = [m for m in markers if '-l' in m]
        shorts = [m for m in markers if '-s' in m]
        epochs_long = epochs[longs]
        epochs_short = epochs[shorts]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_long.get_data(), epochs_short.get_data()])
        y = np.concatenate([np.zeros(len(epochs_long)), np.ones(len(epochs_short))])

    elif condition == 'direction':
        # Get condition:
        ups = [m for m in markers if 'BT' in m]
        downs = [m for m in markers if 'TT' in m]
        lefts = [m for m in markers if 'RT' in m]
        rights = [m for m in markers if 'LT' in m]
        epochs_up = epochs[ups]
        epochs_down = epochs[downs]
        epochs_right = epochs[rights]
        epochs_left = epochs[lefts]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_up.get_data(), epochs_down.get_data(), epochs_right.get_data(), epochs_left.get_data()])
        y = np.concatenate([np.zeros(len(epochs_up)), np.ones(len(epochs_down)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left))])

    elif condition == 'direction_short':
        ups = [m for m in markers if 'BTT-s' in m]
        downs = [m for m in markers if 'TTB-s' in m]
        lefts = [m for m in markers if 'RTL-s' in m]
        rights = [m for m in markers if 'LTR-s' in m]
        epochs_up = epochs[ups]
        epochs_down = epochs[downs]
        epochs_right = epochs[rights]
        epochs_left = epochs[lefts]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_up.get_data(), epochs_down.get_data(), epochs_right.get_data(), epochs_left.get_data()])
        y = np.concatenate([np.zeros(len(epochs_up)), np.ones(len(epochs_down)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left))])

    elif condition == 'direction_long':
        ups = [m for m in markers if 'BTT-l' in m]
        downs = [m for m in markers if 'TTB-l' in m]
        lefts = [m for m in markers if 'RTL-l' in m]
        rights = [m for m in markers if 'LTR-l' in m]
        epochs_up = epochs[ups]
        epochs_down = epochs[downs]
        epochs_right = epochs[rights]
        epochs_left = epochs[lefts]

        # Create data matrix X (epochs x channels x timepoints) and label vector y (epochs x 1):
        X = np.concatenate([epochs_up.get_data(), epochs_down.get_data(), epochs_right.get_data(), epochs_left.get_data()])
        y = np.concatenate([np.zeros(len(epochs_up)), np.ones(len(epochs_down)), 2*np.ones(len(epochs_right)), 3*np.ones(len(epochs_left))])

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    n_len = X.shape[2] - n_timepoints + 1
    for tp in range(n_timepoints, X.shape[2]+1):
        for ch in range(X.shape[1]):
            x = X[:,ch,tp-n_timepoints:tp]
            # x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
            print(x.shape, end='\r')
            scores = cross_val_score(clf, x, y, cv=5, n_jobs=-2)

            # Add row to the dataframe:
            row_to_add = {'Timepoint': (tp-1)/10 + epochs.tmin, 'Accuracy': scores.mean(), 'Subject': sbj,
                          'N_timepoints': ch, 'Type': epoch_type, 'Init_marker': [markers],
                          't_min': epochs.tmin, 't_max': epochs.tmax, 'epoch_info': [epochs.info],
                          'Date':datetime.now().strftime('%Y-%m-%d'), 'Time': datetime.now().strftime('%H:%M:%S'),
                          'Condition': condition}
            df_scores = pd.concat([df_scores, pd.DataFrame(row_to_add)], ignore_index=True)

        if tp != n_len:
            print(f'Measuring timestamp {tp}/{n_len}', end='\r')
        else:
            print(f'Measuring timestamp {tp}/{n_len}')

    # Add mean of scores as subject: Mean:
    # row_to_add = {'Timepoint': (np.arange(n_timepoints-1, X.shape[2])/10 + epochs.tmin).tolist(),
    #               'Accuracy': df_scores.groupby('Timepoint')['Accuracy'].mean().to_list(),
    #               'Subject': ['Mean']*n_len, 'N_timepoints': [n_timepoints]*n_len, 'Type': [epoch_type]*n_len,
    #               'Init_marker': [markers]*n_len, 't_min': [epochs.tmin]*n_len, 't_max': [epochs.tmax]*n_len,
    #               'epoch_info': [[epochs.info]]*n_len, 'Date':[datetime.now().strftime('%Y-%m-%d')]*n_len,
    #               'Time': [datetime.now().strftime('%H:%M:%S')]*n_len, 'Condition': [condition]*n_len]}

    # df_scores = pd.concat([df_scores, pd.DataFrame(row_to_add)], ignore_index=True)

    # Store dataframe to full classification dataframe:
    _store_scores_df(df_scores, csv_name='dataframes/classification/classification_single_channel_df.csv')




