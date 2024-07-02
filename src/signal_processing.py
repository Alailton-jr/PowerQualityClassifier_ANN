
import numpy as np, librosa, pywt, emd, os

if os.name == 'posix': 
    # Stockwell Package only avaliable for Linux because of FFTW Library
    from stockwell import st


def compute_stft(data, n_fft=128, hop_length=1):
    '''
    Compute Short-Time Fourier Transform of the data using librosa package

    Parameters:
    data: np.array, shape (n_data, n_sample), input data
    n_fft: int, number of FFT
    hop_length: int, hop length for the STFT

    Returns:
    data_stft: np.array, shape (n_data, n_features_N, n_features_M), STFT of the data
    '''
    if len(data.shape) == 1:
        data = np.expand_dims(data, 0)
    data_stft = librosa.stft(data[0], n_fft=n_fft, hop_length=hop_length)[:15, ::1]
    data_stft = np.zeros((data.shape[0], data_stft.shape[0], data_stft.shape[1]))
    for i in range(data.shape[0]):
        data_stft[i] = np.abs(librosa.stft(data[i], n_fft=128, hop_length=hop_length)[:15, ::1])
    return data_stft

def compute_cwt(data, scales=(1,10), df=60):
    '''
    Compute Continuous Wavelet Transform of the data using PyWavelets package

    Parameters:
    data: np.array, shape (n_data, n_sample), input data
    scales: tuple, scales for the CWT
    df: int, frequency resolution

    Returns:
    data_cwt: np.array, shape (n_data, n_features_N, n_features_M), CWT of the data
    '''
    if len(data.shape) == 1:
        data = np.expand_dims(data, 0)
    f = pywt.scale2frequency('mexh', np.arange(*scales))/(1/df)
    data_cwt, _ = pywt.cwt(data, f, 'mexh')
    data_cwt = np.transpose(data_cwt, (1, 0, 2))
    data_cwt = data_cwt[:, :, ::2]
    return data_cwt

def compute_st(data, fmin=0, fmax=1800, df=60, gamma=0.2, win_type='gauss'):
    '''
    Compute Stockwell Transform of the data using Stockwell package

    Parameters:
    data: np.array, shape (n_data, n_sample), input data
    fmin: int, minimum frequency
    fmax: int, maximum frequency
    df: int, frequency resolution
    gamma: float, gamma parameter for the Stockwell Transform
    win_type: str, window type for the Stockwell Transform

    Returns:
    data_st: np.array, shape (n_data, n_features_N, n_features_M), ST of the data
    '''
    if os.name != 'posix':
        raise Exception("This function is not available in this OS")
    if len(data.shape) == 1:
        data = np.expand_dims(data, 0)
    fmin_samples = int(fmin/df)
    fmax_samples = int(fmax/df)
    data_st = np.zeros((data.shape[0], fmax_samples - fmin_samples+1, data.shape[1]))
    for i in range(data.shape[0]):
        data_st[i, :, :] = np.abs(st.st(data[i, :], lo=fmin_samples, hi=fmax_samples, gamma=gamma, win_type=win_type))
    return data_st

vectorized_function = np.vectorize(
    lambda x, freq_range, sum_time: emd.spectra.hilberthuang(x[1], x[2], freq_range, sum_time)[1],
    signature='(n, m, o)->(100,256)',
    excluded=['freq_range', 'sum_time']
)

def compute_hht(data, fs = 15360, freq_range=(0,500,100)):
    '''
    Compute Hilbert-Huang Transform of the data using EMD package

    Parameters:
    data: np.array, shape (n_data, n_sample), input data
    fs: int, sampling frequency
    freq_range: tuple, frequency range for the HHT

    Returns:
    data_hht: np.array, shape (n_data, n_features_N, n_features_M), HHT of the data
    '''
    if len(data.shape) == 1:
        data = np.expand_dims(data, 0)
    Is = np.apply_along_axis(emd.spectra.frequency_transform, -1, data, fs, 'hilbert')
    data_hht = vectorized_function(Is, freq_range=freq_range, sum_time=False)
    return data_hht
