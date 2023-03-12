import os
import mne
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, LeaveOneOut
from datetime import datetime, timezone


def classify(src, dst, sbj, condition, n_timepoints=1):
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
        x = X[:,:,tp-n_timepoints:tp+1]
        x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))

        scores = cross_val_score(clf, x, y, cv=LeaveOneOut(), n_jobs=-1)

        # Add row to the dataframe:
        row_to_add = {'Timepoint': (tp-1)/10 + epochs.tmin, 'Accuracy': scores.mean(), 'Subject': sbj,
                      'N_timepoints': n_timepoints, 'Type': epoch_type, 'Init_marker': [markers],
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
    _store_scores_df(df_scores, csv_name='dataframes/classification/classification_df.csv')


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





