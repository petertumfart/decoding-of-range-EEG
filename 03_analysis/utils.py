import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import exists
from importlib import reload
import numpy as np
import pandas as pd
import pyxdf
import mne
from utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
import time
import datetime
from datetime import datetime, timezone
import pickle
import plotly.express as px

def split_streams(streams):
    """Seperate the streams from the xdf file into an EEG-stream and a Markers stream.

    Args:
        streams (list): List with len=2 that contains the eeg and the marker streams.

    Returns:
        eeg_stream: Stream containing the EEG data.
        marker_stream: Stream containing the Markers.
    """

    assert len(streams) == 2, f'Length should be 2, got {len(streams)}'
        
    for s in streams:
        if s['info']['type'][0] == 'EEG':
            eeg_stream = s
        elif s['info']['type'][0] == 'Marker':
            marker_stream = s
    
    return eeg_stream, marker_stream


def plot_spectrum(eeg_struct):
    N = eeg_struct.n_times
    T = 1.0 / eeg_struct.info['sfreq']
    y, times = eeg_struct[0, :]
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    y_plot = 2.0/N * np.abs(yf[:, :N//2])
    plt.plot(xf, y_plot[0, :])
    plt.show()


# Helper functions:
def extract_eeg(stream, kick_last_ch=True):
    """
    Extracts the EEG data and the EEG timestamp data from the stream and stores it into two lists.
    :param stream: XDF stream containing the EEG data.
    :param kick_last_ch: Boolean to kick out the brainproducts marker channel
    :return: eeg: list containing the eeg data
             eeg_ts: list containing the eeg timestamps.cd
    """
    extr_eeg = stream['time_series'].T
    extr_eeg *= 1e-6 # Convert to volts.
    assert extr_eeg.shape[0] == 65
    extr_eeg_ts = eeg_stream['time_stamps']

    if kick_last_ch:
        # Kick the last row (unused Brainproduct markers):
        extr_eeg = extr_eeg[:64,:]

    return extr_eeg, extr_eeg_ts


def extract_eeg_infos(stream):
    """
    Takes eeg stream and extracts the sampling rate, channel names, channel labels and the effective sample rate from the xdf info.
    :param stream: EEG xdf stream
    :return: sampling_rate: Configured sampling rate
    :return: names: channel names
    :return: labels: channel labels (eeg or eog)
    :return: effective_sample_frequency: Actual sampling frequency based on timestamps.
    """
    # Extract all infos from the EEG stream:
    recording_device = stream['info']['name'][0]
    sampling_rate = float(stream['info']['nominal_srate'][0])
    effective_sample_frequency = float(stream['info']['effective_srate'])

    # Extract channel names:
    chn_names = [stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(64)]
    # chn_names.append('Markers')
    labels = ['eeg' for i in range(64)]
    labels[16] = 'eog'
    labels[21] = 'eog'
    labels[40] = 'eog'
    # chn_labels.append('misc')

    return sampling_rate, chn_names, labels, effective_sample_frequency


def extract_annotations(mark_stream, first_samp):
    """
    Function to extract the triggers of the marker stream in order to prepare for the annotations.
    :param mark_stream: xdf stream containing the markers and time_stamps
    :param first_samp: First EEG sample, serves for aligning the markers
    :return: triggs: Dict containing the extracted triggers.
    """
    triggs = {'onsets': [], 'duration': [], 'description': []}

    # Extract the markers:
    marks = mark_stream['time_series']

    # Fix markers due to bug in paradigm:
    corrected_markers = fix_markers(marks)

    # Extract the timestamp of the markers and correct them to zero
    marks_ts = mark_stream['time_stamps'] - first_samp

    # Read every trigger in the stream
    for index, marker_data in enumerate(corrected_markers):
        # extract triggers information
        triggs['onsets'].append(marks_ts[index])
        triggs['duration'].append(int(0))
        # print(marker_data[0])
        triggs['description'].append(marker_data[0])

    return triggs

# Fix markers:
def fix_markers(orig_markers):
    """
    Given a list of markers, this function processes the markers and modifies the trial type markers if necessary.
    Due to shuffle-bug in the paradigm.

    :param orig_markers: A list of markers. Each marker is a tuple containing the marker string and a float value representing the time at which the marker occurred.
    :type orig_markers: list
    :return: The modified list of markers.
    :rtype: list
    """

    trial_type_markers = ['LTR-s', 'LTR-l','RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    counter_letter = {'l': 'R', 'r': 'L', 'b': 'T', 't': 'B'}

    # Parse through markers
    for i in range(len(orig_markers)-3):
        marker = orig_markers[i][0]
        if marker in trial_type_markers:
            following_markers = []
            # Find the next 4 occurances that start with 'c':
            # and store them in a list:
            if (i+9) < len(orig_markers):
                for ii in range(i+1, i+9):
                    next_mark = orig_markers[ii][0]
                    if next_mark[0] == 'c':
                        following_markers.append(next_mark[2])
            else:
                for ii in range(i+1, len(orig_markers)):
                    next_mark = orig_markers[ii][0]
                    if next_mark[0] == 'c':
                        following_markers.append(next_mark[2])

            # Exit loop if less than 4 following markers were found:
            if len(following_markers) < 4:
                continue

            if following_markers[0] == 'c' or following_markers[1] == 'c':
                continue

            # Extract first letter of the trial type marker:
            first_letter = marker[0].lower()
            last_letter = marker[-1].lower()

            # Check if the first two letters in following markers are the same, if not, change type:
            if (following_markers[0] != first_letter) and (following_markers[1] != first_letter):
                # Trial type changes:
                new_type = following_markers[0].upper() + 'T' + counter_letter[following_markers[0]] + '-'

                if (following_markers[2] == 'c') and (following_markers[3] == 'c'):
                    new_type = new_type + 's'
                else:
                    new_type = new_type + 'l'

                orig_markers[i][0] = new_type

            # Otherwise check if the second two markers are short or long and change accordingly:
            else:
                if (last_letter == 's') and (following_markers[2] != 'c') and (following_markers[3] != 'c'):
                    new_type = marker[:-1]
                    new_type += 'l'
                    orig_markers[i][0] = new_type

                elif (last_letter == 'l') and (following_markers[2] == 'c') and (following_markers[3] == 'c'):
                    new_type = marker[:-1]
                    new_type += 's'
                    orig_markers[i][0] = new_type


    return orig_markers

def add_bad_channel_to_df(bad_chn_row, ch_names, csv_name='bad_channels.csv'):
    """
    Add a row to a CSV file containing information about bad channels in some data.

    :param bad_chn_row : list
        A list containing the information to be added to the CSV file. The order of the elements should
        match the order of the columns in the CSV file.
    :param csv_name : str, optional
        The name of the CSV file. The default is 'bad_channels.csv'.
    :return: df_bads : pandas.DataFrame
        A dataframe containing the information from the CSV file, with the new row added.
    """
    # Check if df_bads.csv already exists:
    if not exists(csv_name):
        # Create dataframe with bad channels:
        df_bads = pd.DataFrame(columns=['Subject', 'Run', 'Paradigm', 'Bad_channel'])
        df_bads.to_csv(csv_name)
    else:
        # Load dataframe
        df_bads = pd.read_csv(csv_name, index_col=0)

    # Check if the channel name exists:
    if bad_chn_row[-1] not in ch_names:
        raise NameError('Channel name not found')

    # Add row to the dataframe:
    df_bads.loc[len(df_bads.index)] = bad_chn_row

    print(f'Added {bad_chn_row} to the dataframe...')

    # Drop duplicates:
    df_bads.drop_duplicates(inplace=True)

    # Save df:
    df_bads.to_csv(csv_name)

    return df_bads

def add_bad_epoch_to_df(bad_epoch_row, csv_name='bad_epochs.csv'):
    # Check if bad_epochs.csv already exists:
    if not exists(csv_name):
        # Create dataframe with bad channels:
        df_bads = pd.DataFrame(columns=['Subject', 'Paradigm', 'Bad epoch'])
        df_bads.to_csv(csv_name)
    else:
        # Load dataframe
        df_bads = pd.read_csv(csv_name, index_col=0)

    # Add row to the dataframe:
    df_bads.loc[len(df_bads.index)] = bad_epoch_row

    print(f'Added {bad_epoch_row} to the dataframe...')

    # Drop duplicates:
    df_bads.drop_duplicates(inplace=True)

    # Save df:
    df_bads.to_csv(csv_name)

    return df_bads



def get_bads_for_subject(subject, csv_file='bad_channels.csv'):
    """
    Get a list of bad channels that appear more than once for a given subject from a CSV file.

    :param subject: Subject name.
    :type subject: str
    :param csv_file: CSV file containing bad channel information. Default is 'bad_channels.csv'.
    :type: csv_file: str

    :returns: list: List of bad channels that appear more than once.

    :raises: FileExistsError: If the CSV file does not exist.
    """
    # Check if df_bads.csv already exists:
    if not exists(csv_file):
        raise FileExistsError('File does not exist, please use the add_bad_channel_df() function.')
    else:
        # Load dataframe
        df = pd.read_csv(csv_file, index_col=0)

    # Filter for subject and check if channel has more then 1 appearances:
    subject_df = df[df['Subject'] == subject]

    # Get the counts of all the unique values in the 'column_name' column
    channel_counts = subject_df['Bad_channel'].value_counts()

    # Select the rows that have a count greater than 1
    duplicate_bads = list(channel_counts[channel_counts>1].index)

    return duplicate_bads

def get_all_additional_information(subject, csv_file='participant_info.csv'):
    """Returns a tuple of additional information for the given subject.

    :param subject: The name of the subject.
    :type subject: str
    :param csv_file: The file path to the participant info CSV file.
    :type csv_file: str
    :return: A tuple containing the following information:
        - meas_date (datetime): The measurement date.
        - experimenter (str): The name of the experimenter.
        - proj_name (str): The name of the project.
        - subject_info (str): The name of the subject.
        - line_freq (float): The line frequency.
        - gender (str): The gender of the subject.
        - dob (str): The date of birth of the subject.
        - age_at_meas (float): The age of the subject at the time of measurement.
    :rtype: tuple
    """
    if not isinstance(subject, str):
        raise TypeError('Subject must be a string.')
    if not isinstance(csv_file, str):
        raise TypeError('CSV file must be a string.')
    if not exists(csv_file):
        raise FileNotFoundError('File does not exist. Check if the path is correct.')

    df = pd.read_csv(csv_file, index_col=False)
    subject_info = df[df['Participant'] == subject]

    if subject_info.empty:
        raise ValueError('Subject not found in CSV file.')

    meas_date_str = subject_info['Measurement_Date'].values[0]
    meas_date = datetime.strptime(meas_date_str, '%d.%m.%Y')
    meas_date = meas_date.replace(tzinfo=timezone.utc)
    experimenter = 'Peter T.'
    proj_name = 'Decoding of range during goal-directed movement'
    line_freq = 50.0
    gender = subject_info['Gender'].values[0]
    dob = subject_info['Date_Of_Birth'].values[0]
    age_at_meas = subject_info['Age_At_Measurement'].values[0]

    return meas_date, experimenter, proj_name, subject_info, line_freq, gender, dob, age_at_meas

def get_subset_of_dict(full_dict, keys_of_interest):
    return dict((k, full_dict[k]) for k in keys_of_interest if k in full_dict)


def create_sliced_trial_list(event_dict, events_from_annot):
    # Slice into list of list from trial_type_marker to trial_type_marker
    trial_type_markers = ['LTR-s', 'LTR-l','RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    event_dict_trial_type = get_subset_of_dict(event_dict, trial_type_markers)
    event_sequence = events_from_annot[:,-1]

    trial_list = []
    first_samps = []
    first_time = True
    for i, entry in enumerate(event_sequence):
        if entry in event_dict_trial_type.values():
            if first_time:
                temp_list = [entry]
                first_samps.append(events_from_annot[i,0])
                first_time = False
            else:
                temp_list.append(entry)
                trial_list.append(temp_list)
                temp_list = [entry]
                first_samps.append(events_from_annot[i,0])
        else:
            if not first_time:
                temp_list.append(entry)

    trial_list.append(temp_list)

    return trial_list, first_samps


def get_bad_epochs(event_dict, trial_list):
    """
    Given an event dictionary, find the indices of the epochs (sub-lists) in the trial list that are invalid.
    An epoch is invalid if it does not satisfy the following conditions:
        1. If it is not the last epoch, its length must be 9.
        2. If it is the last epoch, its length must be 8.
        3. The first entry must be a trial_type marker.
        4. The second entry must be the 'Start' marker.
        5. The fourth entry must be the 'Cue' marker.
        6. The seventh entry must be the 'Break' marker.
        7. The first two LDR readings must be coherent with the trial type.
        8. The second two LDR readings must be coherent with the trial type.

    :param event_dict: A dictionary where keys are event names and values are corresponding event markers.
    :type event_dict: dict
    :return: A list of indices corresponding to the invalid epochs.
    :rtype: list
    """

    # Check if the order is correct:
    bad_idcs = []
    trial_type_markers = ['LTR-s', 'LTR-l','RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    trial_vals = [event_dict[key] for key in trial_type_markers]
    n_epochs = len(trial_list)

    for idx, sub_list in enumerate(trial_list):
        # Add bad epoch if the length is not 9 (except for the last epoch):
        if len(sub_list) != 9 and idx != n_epochs-1:
            bad_idcs.append(idx)
            continue

        # Add bad epoch if the length is not 8 for the last epoch:
        elif len(sub_list) != 8 and idx == n_epochs-1:
            bad_idcs.append(idx)
            continue

        # Add bad epoch if the first entry is not a trial_type_marker:
        if sub_list[0] not in trial_vals:
            bad_idcs.append(idx)
            continue

        # Add bad epoch if the second entry is not a Start marker:
        if sub_list[1] != event_dict['Start']:
            bad_idcs.append(idx)
            continue

        # Add bad epoch if the fourth entry is not a Cue marker:
        if sub_list[3] != event_dict['Cue']:
            bad_idcs.append(idx)
            continue

        # Add bad epoch if the seventh entry is not a Break marker:
        if sub_list[6] != event_dict['Break']:
            bad_idcs.append(idx)
            continue

        # Get the keys for entries 3,5,6 and 8:
        start_touch = list(event_dict.keys())[list(event_dict.values()).index(sub_list[2])]
        start_release = list(event_dict.keys())[list(event_dict.values()).index(sub_list[4])]
        target_touch = list(event_dict.keys())[list(event_dict.values()).index(sub_list[5])]
        target_release = list(event_dict.keys())[list(event_dict.values()).index(sub_list[7])]

        # Get key for the trial_type marker:
        trial_type = list(event_dict.keys())[list(event_dict.values()).index(sub_list[0])]

        # Add bad epoch if first two ldr readings are not coherent with the trial type:
        if (trial_type[0].lower() != start_touch[2]) or (trial_type[0].lower() != start_release[2]):
            bad_idcs.append(idx)
            continue

        # Add bad epoch if the second two ldr readings are not coherent with the second part of the trial type:
        if (trial_type[4] == 'l'):
            if (trial_type[2].lower() != target_touch[2]) or (trial_type[2].lower() != target_release[2]):
                bad_idcs.append(idx)
                continue

        if (trial_type[4] == 's'):
            if (target_touch[2] != 'c') or (target_release[2] != 'c'):
                bad_idcs.append(idx)
                continue

    return bad_idcs

def convert_samps_to_time(first_time, first_samp, samp_list):
    """Convert sample numbers to time values.
    :param first_time: float time value of the first sample
    :param first_samp: int sample number of the first sample
    :param samp_list: list of int sample numbers to be converted
    :return: numpy ndarray of time values for the input sample numbers
    """
    return np.array(samp_list) * first_time / first_samp

def create_bad_annotations(starting_times, bad_events, duration, orig_time):
    """Create annotations for bad events in EEG data.

    :param starting_times: 1D array of starting times for all events in EEG data
    :type starting_times: numpy.ndarray
    :param bad_events: Indices of bad events in the starting_times array
    :type bad_events: numpy.ndarray or list
    :param duration: Duration of the bad events
    :type duration: float
    :param orig_time: The time at which the first sample in data was recorded
    :type orig_time: float
    :return: mne.Annotations object containing onsets, durations, and descriptions for bad events
    :rtype: mne.Annotations
    """

    bad_times = starting_times[bad_events]
    onsets = bad_times + 0.01
    durations = [duration] * len(bad_times)
    descriptions = ['bad epoch'] * len(bad_times)
    return mne.Annotations(onsets, durations, descriptions, orig_time=orig_time)

def rename_annotations(descriptions):
    """
        Rename the annotations of touch/release markers in the form of
        new_marker = trial_type + period + position + state
        where trial_type e.g. 'LTR-l'
        period is either 'i' (indication) or 'c' (cue)
        position is the position from the marker e.g. the 't' from c t 0
        state is the touch or release state from the marker e.g. for c t 0 the state is '0' (touch). '1' would be release.

        :param descriptions: list of strings, annotations to rename
        :return: list of strings, renamed annotations
    """

    trial_type_markers = ['LTR-s', 'LTR-l','RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    for i, entry in enumerate(descriptions):
        if entry in trial_type_markers:
            if 'bad' in descriptions[i+1]:
                continue
            else:
                trial_type = entry
                period = 'i' # indication
                position = descriptions[i+2][2]
                state = descriptions[i+2][4]

                descriptions[i+2] = trial_type + '_' + period + position + state

                trial_type = entry
                period = 'i' # indication
                position = descriptions[i+4][2]
                state = descriptions[i+4][4]

                descriptions[i+4] = trial_type + '_' + period + position + state

                trial_type = entry
                period = 'c' # cue
                position = descriptions[i+5][2]
                state = descriptions[i+5][4]

                descriptions[i+5] = trial_type + '_' + period + position + state

                trial_type = entry
                period = 'c' # cue
                position = descriptions[i+7][2]
                state = descriptions[i+7][4]

                descriptions[i+7] = trial_type + '_' + period + position + state

    return descriptions

def generate_markers_of_interest(trial_type, period, position, state):
    """
    Generates markers of interest based on provided parameters

    :param trial_type: A list of trial types
    :type trial_type: list
    :param period: A list of periods
    :type period: list
    :param position: A list of positions
    :type position: list
    :param state: A list of states
    :type state: list
    :return: A list of markers of interest
    :rtype: list
    """
    moi = []
    for tp in trial_type:
        for per in period:
            for pos in position:
                for s in state:
                    moi.append(tp + '_' + per + pos + s)
    return moi

def replace_markers(m_stream, file):
    fname = file[:-4] + '_cleaned.csv'
    df_cleaned = pd.read_csv(fname)

    time_series_cleaned = np.array(df_cleaned.time_series).tolist()
    time_series_cleaned = [[mark] for mark in time_series_cleaned]

    time_stamps_cleaned = np.array(df_cleaned.time_stamps)

    m_stream['time_series'] = time_series_cleaned
    m_stream['time_stamps'] = time_stamps_cleaned

    return m_stream

def store_scores_df(df_to_append, csv_name='classification_df.csv'):
    # Check if dataframe exists and if not, create it:
    if not exists(csv_name):
        df_to_append.to_csv(csv_name)
        return df_to_append
    else:
        df = pd.read_csv(csv_name, index_col=0)
        df = pd.concat([df, df_to_append], ignore_index=True)
        df.to_csv(csv_name)
        return df



def create_scores_df():
    # Create dataframe for storing all the classification data + information:
    cols = ['Timepoint',   # Timepoint of classification accuracy
            'Accuracy',    # Classification accuracy
            'Subject',     # Subject ID
            '5-point',     # True or False based on classification approach
            'Type',        # Cue-aligned or Movement onset-aligned
            'Init_marker', # Marker(s) used for epoching
            't_min',
            't_max',
            'epoch_info',
            'Date',
            'Time']

    df = pd.DataFrame(columns=cols)
    df['5-point'] = df['5-point'].astype(bool)

    return df

def load_raw_file(dirpath, file):
    file = dirpath + '/' + file
    return mne.io.read_raw(file, preload=True)

def create_parameter_matrix(epochs):
    """
    Creates a parameter matrix for the given epochs.

    :param epochs: A list of MNE-Python Epochs objects.
                   Each Epochs object represents a segment of EEG data.
    :type epochs: list of mne.Epochs objects

    :return: The parameter matrix for the given epochs.
             The matrix has shape (5, N), where N is the number of epochs.
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

    # Return S matrix:
    return np.concatenate((s_short, s_long, s_vert, s_horz, s_intercept),axis=0)