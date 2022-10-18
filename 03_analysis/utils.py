import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt

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
