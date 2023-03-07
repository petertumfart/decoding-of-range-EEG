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