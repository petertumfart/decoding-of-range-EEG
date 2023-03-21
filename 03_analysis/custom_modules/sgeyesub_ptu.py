import os
import mne
import pandas as pd
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
