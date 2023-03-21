import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timezone
from scipy.stats import t


def calc_eye_derivations(src, dst, sbj):
    """
    Calculates horizontal, vertical, and radial EOG derivatives from raw EOG data and
    saves the resulting raw object to a file.

    :param src: The directory path of the source files containing the raw EOG data.
    :type src: str
    :param dst: The directory path where the resulting raw object with EOG derivatives will be saved.
    :type dst: str
    :param sbj: The subject identifier used to match the EOG file to be processed.
    :type sbj: str
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and ('eye' in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Extract the EOG data:
    eog_r = raw.get_data(picks='EOGL')
    eog_l = raw.get_data(picks='EOGR')
    eog_c = raw.get_data(picks='EOGC')

    # Calculate EOG derivatives:
    heog = eog_r - eog_l
    veog = eog_c - (eog_r + eog_l) / 2
    reog = (eog_c + eog_r + eog_l) / 3

    # Create nchannels x ntimes matrix:
    eog_derivatives = np.vstack((heog, veog, reog))

    # Create an info file to later add the channels:
    info = mne.create_info(['EOGH', 'EOGV', 'EOGRad'], sfreq=raw.info['sfreq'], ch_types='eog')

    # Create a raw array for the eog derivatives:
    raw_derivatives = mne.io.RawArray(eog_derivatives, info, first_samp=raw.first_samp)

    # Append the new raw object to the existing raw object:
    raw = raw.add_channels([raw_derivatives], force_update_info=True)

    # Store the filtered file:
    store_name = dst + '/' + sbj + '_' + 'eye' + '_eog_deriv_added_raw.fif'
    raw.save(store_name, overwrite=True)


def lp_filter_derivatives(src, dst, sbj):
    """
    This function applies a lowpass filter to the EOG channels of a raw MNE file and saves the filtered file
    with a new name in a specified directory.

    :param src: A string representing the source directory containing the raw MNE file.
    :param dst: A string representing the destination directory where the filtered MNE file will be saved.
    :param sbj: A string representing the subject name used to identify the raw and filtered MNE files.
    """

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and ('eye' in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Lowpass filter:
    raw = raw.copy().filter(l_freq=None, h_freq=5.0, picks=['eog'], method='iir')

    # Store the filtered file:
    store_name = dst + '/' + sbj + '_' + 'eye' + '_lp_filtered_raw.fif'
    raw.save(store_name, overwrite=True)


def epoch_eye_paradigm(src, dst, sbj, mrks):
    """
   Epoch the EEG data for a given subject in the eye paradigm.

   :param src: The path to the directory containing the raw EEG data.
   :type src: str
   :param dst: The path to the directory where the epoched data will be stored.
   :type dst: str
   :param sbj: The subject identifier to be used in the output file name.
   :type sbj: str
   :param mrks: A dictionary containing the markers of interest and their corresponding IDs.
   :type mrks: list
   """

    mne.set_log_level('INFO')

    # There can be only one file  with matching conditions since we are splitting in folders:
    f_name = [f for f in os.listdir(src) if (sbj in f) and ('eye' in f)][0]

    file = src + '/' + f_name
    raw = mne.io.read_raw(file, preload=True)

    # Extract epochs:
    events_from_annot, event_dict = mne.events_from_annotations(raw)

    # Get event dict:
    event_dict_of_interest = _get_subset_of_dict(event_dict, mrks)

    # Epoch the data:
    epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict_of_interest, tmin=1.0, tmax=9.0,
                        baseline=None, reject_by_annotation=True, preload=True, picks=['eeg', 'eog'])

    # Store the epoched file:
    store_name = dst + '/' + sbj + '_' + 'eye' + '_epoched_for_eyesub.fif'
    epochs.save(store_name, overwrite=True)

    mne.set_log_level('WARNING')


def _get_subset_of_dict(full_dict, keys_of_interest):
    return dict((k, full_dict[k]) for k in keys_of_interest if k in full_dict)
