import os
import mne

def concat_fifs(src, dst, sbj, paradigm='paradigm'):
    """
    Concatenates multiple raw.fif files from a single subject and paradigm.

    :param src: str, path to the directory containing the input raw.fif files.
    :param dst: str, path to the directory where the concatenated raw.fif file will be stored.
    :param sbj: str, subject name contained in the file names.
    :param paradigm: str, paradigm name contained in the file names (default is 'paradigm').
    :return: None

    - Searches for all the files in the source directory containing the subject name, 'raw.fif' and the specified paradigm.
    - Reads each of the selected raw.fif files using MNE-Python's read_raw function.
    - Concatenates the read files into a single raw object using MNE-Python's concatenate_raws function.
    - Saves the concatenated raw object as a new raw.fif file in the destination directory.
    """
    file_names = [f for f in os.listdir(src) if (sbj in f) and ('raw.fif' in f) and (paradigm in f)]

    raws = []
    for i, f_name in enumerate(file_names):
        print(f'#', end=' ')

        file = src + '/' + f_name
        raw = mne.io.read_raw(file, preload=True)
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
    raw = raw.copy().filter(l_freq=0.4, h_freq=None, picks=['eeg'], method='iir')

    # Notch filter:
    raw = raw.copy().notch_filter(freqs=[50], picks=['eeg'])

    # Store the filtered file:
    store_name = dst + '/' + sbj + '_' + paradigm + '_highpass_notch_filtered_raw.fif'
    raw.save(store_name, overwrite=True)