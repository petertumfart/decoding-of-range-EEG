import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timezone
from scipy.stats import t
import scipy.io


def concat_fifs(src, dst, sbj, paradigm='paradigm'):
    """
    Concatenates multiple raw.fif files from a single subject and paradigm.

    :param src: str, path to the directory containing the input raw.fif files.
    :param dst: str, path to the directory where the concatenated raw.fif file will be stored.
    :param sbj: str, subject name contained in the file names.
    :param paradigm: str, paradigm name contained in the file names (default is 'paradigm').
    :return: None

    - Searches for all the files in the source directory containing the subject name, 'raw.fif' and the specified
     paradigm.
    - Reads each of the selected raw.fif files using MNE-Python's read_raw function.
    - Concatenates the read files into a single raw object using MNE-Python's concatenate_raws function.
    - Saves the concatenated raw object as a new raw.fif file in the destination directory.
    """
    file_names = [f for f in os.listdir(src) if (sbj in f) and ('raw.fif' in f) and (paradigm in f)]

    # Get correct info:
    meas_date, experimenter, proj_name, subject_info, line_freq, gender, dob, age_at_meas = \
        _get_all_additional_information(sbj, csv_file='dataframes/preprocessing/participant_info.csv')

    big_subject_info = {'Subject ID': sbj,
                        'Gender': gender,
                        'Age at measurement': age_at_meas}

    raws = []
    for i, f_name in enumerate(file_names):
        print(f'#', end=' ')

        file = src + '/' + f_name
        raw = mne.io.read_raw(file, preload=True)

        # Add infos:
        raw.info['subject_info'] = big_subject_info
        raw.info['experimenter'] = experimenter
        raw.set_meas_date(meas_date)
        raw.info['line_freq'] = line_freq

        raws.append(raw)

    concat_raw = mne.concatenate_raws(raws)

    # Store the concatenated raw file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_concatenated_raw.fif'
    concat_raw.save(store_name, overwrite=True)


def filter_fifs(src, dst, sbj, paradigm='paradigm'):
    """
    Applies highpass and notch filters to a single raw EEG data file, and saves the filtered data in a new file.

    :param src: The directory path containing the original raw EEG data file.
    :type src: str

    :param dst: The directory path where the filtered EEG data file will be stored.
    :type dst: str

    :param sbj: The subject identifier for the raw EEG data file.
    :type sbj: str

    :param paradigm: The task identifier for the raw EEG data file. Default is 'paradigm'.
    :type paradigm: str

    :return: None
    :rtype: None
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Highpass filter:
    raw = raw.copy().filter(l_freq=0.4, h_freq=None, picks=['eeg', 'eog'], method='iir')

    # Notch filter:
    raw = raw.copy().notch_filter(freqs=[50], picks=['eeg', 'eog'])

    # Store the filtered file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_highpass_notch_filtered_raw.fif'
    raw.save(store_name, overwrite=True)


def _get_bads_for_subject(subject, csv_file='bad_channels.csv'):
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
    if not os.path.exists(csv_file):
        raise FileExistsError('File does not exist, please use the add_bad_channel_df() function.')
    else:
        # Load dataframe
        df = pd.read_csv(csv_file, index_col=0)

    # Filter for subject and check if channel has more then 1 appearances:
    subject_df = df[df['Subject'] == subject]

    # Get the counts of all the unique values in the 'column_name' column
    channel_counts = subject_df['Bad_channel'].value_counts()

    # Select the rows that have a count greater than 0:
    duplicate_bads = list(channel_counts[channel_counts > 0].index)

    return duplicate_bads


def interpolate_bads(src, dst, sbj, paradigm='paradigm'):
    """
    Interpolates bad channels in the raw data file for a given subject and saves the interpolated raw file.

    :param src: The path to the directory containing the raw data files.
    :type src: str
    :param dst: The path to the directory where the interpolated raw file will be saved.
    :type dst: str
    :param sbj: The subject identifier for the data file to be interpolated.
    :type sbj: str
    :param paradigm: The paradigm identifier for the data file to be interpolated. Default is 'paradigm'.
    :type paradigm: str
    :return: None
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Add bad channels:
    bads = _get_bads_for_subject(sbj, csv_file='dataframes/preprocessing/bad_channels.csv')

    # Add the bad channels to the raw.info:
    raw.info['bads'] = bads

    # Interpolate bad channels (based on info:
    raw = raw.copy().interpolate_bads(reset_bads=True)

    # Store the interpolated raw file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_interpolated_raw.fif'
    raw.save(store_name, overwrite=True)

def fit_sgeyesub(src, dst, sbj, paradigm='paradigm'):
    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Load eyesub matrtix:
    C = scipy.io.loadmat(f'dataframes/preprocessing/{sbj}_C.mat')['C']

    raw_subed = raw.copy()
    eeg_channels = mne.pick_types(raw_subed.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # Multiply the custom matrix to EEG channels in the data
    raw_subed._data[eeg_channels, :] = np.dot(C, raw_subed._data[eeg_channels, :])

    # Store the eye artifact subtracted raw file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_eyesubed_raw.fif'
    raw_subed.save(store_name, overwrite=True)


def car(src, dst, sbj, paradigm):
    """
    Apply common average reference (CAR) to the raw EEG data and save it.

    :param src: The source directory containing the raw EEG data files.
    :type src: str

    :param dst: The destination directory where the interpolated raw file will be stored.
    :type dst: str

    :param sbj: The subject identifier string contained in the filename.
    :type sbj: str

    :param paradigm: The task paradigm identifier string contained in the filename.
    :type paradigm: str

    :return: None
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Interpolate bad channels (based on info:
    raw = raw.copy().set_eeg_reference(ref_channels='average')

    # Store the interpolated raw file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_car_raw.fif'
    raw.save(store_name, overwrite=True)


def mark_bad_dataspans(src, dst, sbj, paradigm):
    """
    Mark bad data spans in the raw EEG data and save it.

    :param src: The source directory containing the raw EEG data files.
    :type src: str

    :param dst: The destination directory where the interpolated raw file will be stored.
    :type dst: str

    :param sbj: The subject identifier string contained in the filename.
    :type sbj: str

    :param paradigm: The task paradigm identifier string contained in the filename.
    :type paradigm: str

    :return: None
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Get the events from the raw annotations:
    events_from_annot, event_dict = mne.events_from_annotations(raw)

    # Check if the order of annotations is correct:
    # Therefore first create a marker list of each trial, then convert samples to times and then get the bad events:
    trial_list, starting_samples = _create_sliced_trial_list(event_dict, events_from_annot)
    starting_times = _convert_samps_to_time(raw.first_time, raw.first_samp, starting_samples)
    bad_events = _get_bad_epochs(event_dict, trial_list)
    print(len(bad_events))

    # add annotation for bad channels and select reject_by_annotation when generating the epochs:
    bad_annots = _create_bad_annotations(starting_times, bad_events, duration=7, orig_time=raw.info['meas_date'])
    raw.set_annotations(raw.annotations + bad_annots)

    # Rename annotations to make them unique:
    raw.annotations.description = _rename_annotations(raw.annotations.description)

    # Store the interpolated raw file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_bad_dataspans_marked_raw.fif'
    raw.save(store_name, overwrite=True)


def lowpass_filter(src, dst, sbj, paradigm='paradigm'):
    """
    Applies a low-pass filter to an EEG file.

    :param src: (str) The directory path of the source file.
    :param dst: (str) The directory path where the filtered file will be stored.
    :param sbj: (str) The subject ID.
    :param paradigm: (str) The experiment paradigm (default: 'paradigm').

    :return: None. The filtered file is saved in the specified directory.
    :raises: None.
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Lowpass filter:
    raw = raw.copy().filter(l_freq=None, h_freq=5.0, picks=['eeg', 'eog'], method='iir')

    # Store the filtered file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_lowpass_filtered_raw.fif'
    raw.save(store_name, overwrite=True)


def epoch_for_outlier_detection(src, dst, sbj, paradigm='paradigm'):
    """
    Epochs the EEG data for outlier detection and stores the epoched file.

    :param src: str
        The path to the directory where the raw EEG data is stored.
    :param dst: str
        The path to the directory where the epoched data will be stored.
    :param sbj: str
        The subject identifier.
    :param paradigm: str, optional (default='paradigm')
        The identifier for the paradigm used in the experiment.

    :return: None
    """

    mne.set_log_level('INFO')
    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    events_from_annot, event_dict = mne.events_from_annotations(raw)

    # Define markers of interest
    markers_of_interest = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']

    event_dict_of_interest = _get_subset_of_dict(event_dict, markers_of_interest)

    epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict_of_interest, tmin=0.0, tmax=7.0,
                        baseline=None, reject_by_annotation=True, preload=True, picks=['eeg', 'eog'],
                        reject=dict(eeg=200e-6))

    # Store the epoched file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_epoched_for_outlier_detection_epo.fif'
    epochs.save(store_name, overwrite=True)

    mne.set_log_level('WARNING')


def epoch_and_resample(src, dst, sbj, paradigm='paradigm', cue_aligned=True, resample=True):
    """
    Reads in the raw EEG data from a file, epochs it based on markers of interest, and then downsamples the resulting
    epochs to 10 Hz before saving the result to a new file.

    :param src: str
        The path to the directory where the raw EEG data file is located.
    :param dst: str
        The path to the directory where the epoched and resampled file will be saved.
    :param sbj: str
        The subject identifier to use when creating the file name.
    :param paradigm: str (default='paradigm')
        The paradigm identifier to use when creating the file name.
    :param cue_aligned: bool (default=True)
        A flag indicating whether the epoching should be cue-aligned or indication-aligned.

    :return: None
    """

    mne.set_log_level('INFO')

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and (paradigm in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    events_from_annot, event_dict = mne.events_from_annotations(raw)

    if cue_aligned:
        # Define markers of interest
        markers_of_interest = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']

        event_dict_of_interest = _get_subset_of_dict(event_dict, markers_of_interest)

        epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict_of_interest, tmin=0.0, tmax=7.0,
                            baseline=None, reject_by_annotation=True, preload=True, picks=['eeg'],
                            reject=dict(eeg=200e-6))

    else:
        # Looking at indication release:
        trial_type_markers = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
        period = ['i']  # 'i', 'c' .. indication, cue
        position = ['l', 'r', 't', 'b', 'c']
        state = ['1']  # 0,1 .. touch/release
        markers_of_interest = _generate_markers_of_interest(trial_type_markers, period, position, state)

        event_dict_of_interest = _get_subset_of_dict(event_dict, markers_of_interest)

        epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict_of_interest, tmin=-2.5, tmax=3.5, baseline=None,
                            reject_by_annotation=True, preload=True, picks=['eeg'], reject=dict(eeg=200e-6))

    if resample:
        # Downsample to 10 Hz:
        epochs = epochs.copy().resample(10)

    # Store the epoched file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_epoched_and_resampled_epo.fif'
    epochs.save(store_name, overwrite=True)

    mne.set_log_level('WARNING')


def _get_subset_of_dict(full_dict, keys_of_interest):
    return dict((k, full_dict[k]) for k in keys_of_interest if k in full_dict)


def vis_epochs_for_sbj(src, sbj):
    """
    Reads in the epoched EEG data from a file and returns it.

    :param src: str
        The path to the directory where the epoched EEG data file is located.
    :param sbj: str
        The subject identifier used in the file name.

    :return: mne.Epochs
        The epoched EEG data.
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f)][0]

    file = src + '/' + f_name
    epochs = mne.read_epochs(file, preload=True)

    return epochs


def vis_raw_for_sbj(src, sbj):
    """
    Reads in the raw EEG data from a file and returns it.

    :param src: str
        The path to the directory where the epoched EEG data file is located.
    :param sbj: str
        The subject identifier used in the file name.

    :return: mne.raw
        The raw EEG data.
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f)][0]

    file = src + '/' + f_name
    epochs = mne.io.read_raw(file, preload=True)

    return epochs

def _create_sliced_trial_list(event_dict, events_from_annot):
    """
    Slice the events_from_annot numpy array into a list of trials based on trial type markers.

    :param event_dict: A dictionary with event types as keys and their corresponding values as values.
    :type event_dict: dict

    :param events_from_annot: A numpy array of shape (n_events, 3) with the third column containing the event type.
    :type events_from_annot: numpy.ndarray

    :return: A tuple containing the sliced trial list and the first sample values of each trial.
    :rtype: tuple
    """

    # Slice into list of list from trial_type_marker to trial_type_marker
    trial_type_markers = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    event_dict_trial_type = _get_subset_of_dict(event_dict, trial_type_markers)
    event_sequence = events_from_annot[:, -1]

    trial_list = []
    first_samps = []
    first_time = True
    for i, entry in enumerate(event_sequence):
        if entry in event_dict_trial_type.values():
            if first_time:
                temp_list = [entry]
                first_samps.append(events_from_annot[i, 0])
                first_time = False
            else:
                temp_list.append(entry)
                trial_list.append(temp_list)
                temp_list = [entry]
                first_samps.append(events_from_annot[i, 0])
        else:
            if not first_time:
                temp_list.append(entry)

    trial_list.append(temp_list)

    return trial_list, first_samps


def _convert_samps_to_time(first_time, first_samp, samp_list):
    """Convert sample numbers to time values.
    :param first_time: float time value of the first sample
    :param first_samp: int sample number of the first sample
    :param samp_list: list of int sample numbers to be converted
    :return: numpy ndarray of time values for the input sample numbers
    """
    return np.array(samp_list) * first_time / first_samp


def _get_bad_epochs(event_dict, trial_list):
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
    trial_type_markers = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
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
        if trial_type[4] == 'l':
            if (trial_type[2].lower() != target_touch[2]) or (trial_type[2].lower() != target_release[2]):
                bad_idcs.append(idx)
                continue

        if trial_type[4] == 's':
            if (target_touch[2] != 'c') or (target_release[2] != 'c'):
                bad_idcs.append(idx)
                continue

    return bad_idcs


def _create_bad_annotations(starting_times, bad_events, duration, orig_time):
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


def _rename_annotations(descriptions):
    """
    Rename the annotations of touch/release markers in the form of new_marker = trial_type + period + position +
    state where trial_type e.g. 'LTR-l' period is either 'i' (indication) or 'c' (cue) position is the position from
    the marker e.g. the 't' from c t 0 state is the touch or release state from the marker e.g. for c t 0 the state
    is '0' (touch). '1' would be release.

        :param descriptions: list of strings, annotations to rename
        :return: list of strings, renamed annotations
    """

    trial_type_markers = ['LTR-s', 'LTR-l', 'RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l']
    for i, entry in enumerate(descriptions):
        if entry in trial_type_markers:
            if 'bad' in descriptions[i+1]:
                continue
            else:
                trial_type = entry
                period = 'i'  # indication
                position = descriptions[i+2][2]
                state = descriptions[i+2][4]

                descriptions[i+2] = trial_type + '_' + period + position + state

                trial_type = entry
                period = 'i'  # indication
                position = descriptions[i+4][2]
                state = descriptions[i+4][4]

                descriptions[i+4] = trial_type + '_' + period + position + state

                trial_type = entry
                period = 'c'  # cue
                position = descriptions[i+5][2]
                state = descriptions[i+5][4]

                descriptions[i+5] = trial_type + '_' + period + position + state

                trial_type = entry
                period = 'c'  # cue
                position = descriptions[i+7][2]
                state = descriptions[i+7][4]

                descriptions[i+7] = trial_type + '_' + period + position + state

    return descriptions


def _get_all_additional_information(subject, csv_file='participant_info.csv'):
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
    if not os.path.exists(csv_file):
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


def _generate_markers_of_interest(trial_type, period, position, state):
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

def plot_grand_average(src, dst, sbj_list, paradigm, split=[''], plot_topo=False):

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
            ups = [m for m in markers if 'BTT' in m]
            downs = [m for m in markers if 'TTB' in m]
            lefts = [m for m in markers if 'RTL' in m]
            rights = [m for m in markers if 'LTR' in m]
            combined_conditions = []
            combined_conditions.append(ups)
            combined_conditions.append(downs)
            combined_conditions.append(lefts)
            combined_conditions.append(rights)

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




    # ch_name = 'Cz'
    # idx = [i for i, name in enumerate(epochs.ch_names) if name == ch_name][0]

    x = np.arange(epochs.tmin, epochs.tmax+1/epochs.info['sfreq'], 1/epochs.info['sfreq'])
    for idx, name in enumerate(epochs.ch_names):
        title = f'{title_cond} {title_alignment} {name}'
        legend_text = []
        fig, ax = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3]})
        for i, cond in enumerate(split):
            ax[1].plot(x, grand_avg[i][idx, :]*1e6)
            ax[1].fill_between(x, lowers_l[i][idx, :]*1e6, uppers_l[i][idx, :]*1e6, alpha=0.1)
            legend_text.append(f'Grand_average {cond}')
            legend_text.append('95%-CI')
        # plt.plot(x,grand_avg_short[idx,:]*1e6)
        # plt.fill_between(x, lowers_short[idx,:]*1e6, uppers_short[idx,:]*1e6, alpha=0.1)
        ax[1].plot([t_zero, t_zero], [lowers_l[i][idx, :].min()*1e6, uppers_l[i][idx, :].max()*1e6], color='black')
        legend_text.append( 'Cue presentation')
        ax[1].legend(legend_text, loc='best', prop={'size': 6})
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Voltage (uV)')
        ax[1].set_xlim([x[0], x[-1]])
        fig.suptitle(title)

        if 'cue_aligned' in src:
            ax[0].plot(x[1:-1], smoothed_cue_mov)
            ax[0].plot(x[1:-1], smoothed_cue_fin)
            ax[0].legend(['Release', 'Touch'],
                       prop={'size': 6}, loc='best')

        if 'movement_aligned' in src:
            ax[0].plot(x[1:-1], smoothed_cue_mov)
            ax[0].plot(x[1:-1], smoothed_start_stop)
            ax[0].legend(['Cue', 'Touch'],
                       prop={'size': 6}, loc='best')

        ax[0].set_xlim([x[0], x[-1]])
        plt.savefig(f'{dst}/grand_average_{name}_{title_alignment}_{title_cond}.png', dpi=400)
        plt.close('all')
        # plt.show()




    # plt.plot(range(grand_avg.shape[1]), grand_avg[13,:])
    # plt.show()

    # _calc_grand_average([avg])
    return epochs

def plot_topomaps(src, dst, sbj_list, paradigm, split=[''], times=0.0, ncols=8):
    # Calculate grand average without conditions:
    if split == ['']:
        epochs_lst = [[]]
        full_epochs = []
        title_cond = 'All conditions'

    if split == ['long', 'short']:
        epochs_lst = [[],[]]
        full_epochs = []
        title_cond = 'Distance (long v short)'

    if split == ['up', 'down', 'left', 'right']:
        epochs_lst = [[],[],[],[]]
        full_epochs = []
        title_cond = 'Direction (up v down v left v right)'


    if 'cue_aligned' in src:
        title_alignment = 'cue_aligned'
    if 'movement_aligned' in src:
        title_alignment = 'movement_aligned'

    # Retrieve all filenames from the source directory:
    file_names = [f for f in os.listdir(src)]

    evokeds_lst = []
    for sbj in sbj_list:
        # Should be only one for each subject:
        file = src + '/' + [f for f in file_names if (sbj in f)][0]

        epochs = mne.read_epochs(file, preload=True)

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
            ups = [m for m in markers if 'BTT' in m]
            downs = [m for m in markers if 'TTB' in m]
            lefts = [m for m in markers if 'RTL' in m]
            rights = [m for m in markers if 'LTR' in m]
            combined_conditions = []
            combined_conditions.append(ups)
            combined_conditions.append(downs)
            combined_conditions.append(lefts)
            combined_conditions.append(rights)

        for i, cond in enumerate(split):
            epochs_lst[i].append(epochs[combined_conditions[i]])

    for i, cond in enumerate(split):
        fig = mne.concatenate_epochs(epochs_lst[i]).average().plot_topomap(times, ch_type='eeg',
                                                                           ncols=ncols, nrows='auto',
                                                                           show=False)
        fig.savefig(f'{dst}/topomaps_{title_alignment}_{cond}_{title_cond}.png', dpi=400)


# full_epochs_long.average().plot_topomap(times, ch_type='eeg')
    # full_epochs_short.average().plot_topomap(times, ch_type='eeg')

def _calc_confidence_interval(avg, sbj_list):
    n_chan, n_ts = avg[0].shape
    uppers = np.zeros((n_chan, n_ts))
    lowers = np.zeros((n_chan, n_ts))

    confidence = .95
    n_sample = 50
    for chan in range(n_chan):
        print(chan, end='\r')
        for ts in range(n_ts):
            vals = []
            for subj in range(len(sbj_list)):
                vals.append(avg[subj][chan, ts])

            m = np.array(vals).mean()
            s = np.array(vals).std()
            dof = len(vals)-1

            t_crit = np.abs(t.ppf((1-confidence)/2,dof))

            lowers[chan, ts], uppers[chan, ts] = (m-s*t_crit/np.sqrt(len(vals)), m+s*t_crit/np.sqrt(len(vals)))

    return uppers, lowers

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

    return list(diff_cue_release), list(diff_cue_finished), list(diff_start_stop)

def _gauss(n=55,b=1):
    r = range(-int(n/2),int(n/2)+1)
    return [np.exp(-float(x)**2/(2*b**2)) for x in r]


