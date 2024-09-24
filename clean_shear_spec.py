import numpy as np
from scipy.signal import csd


def clean(A, U, n_fft, rate):
    """
    Remove acceleration contamination from all shear channels.

    Parameters:
    A : ndarray
        Matrix of acceleration signals, with shape (n_samples, n_accel), where each column is an acceleration component.
    U : ndarray
        Matrix of shear probe signals, with shape (n_samples, n_shear), where each column is a shear probe signal.
    n_fft : int
        Length of the FFT used for the calculation of auto- and cross-spectra.
    rate : float
        Sampling rate of the data in Hz.

    Returns:
    clean_UU : ndarray
        Matrix of shear probe cross-spectra after the coherent acceleration signals have been removed.
        The diagonal has the auto-spectra of the cleaned shear signals.
    AA : ndarray
        Matrix of acceleration cross-spectra. The diagonal has the auto-spectra.
    UU : ndarray
        Matrix of shear probe cross-spectra before noise removal.
        The diagonal has the auto-spectra of the original shear signals.
    UA : ndarray
        Matrix of cross-spectra between shear and acceleration signals.
    F : ndarray
        Frequency vector corresponding to the spectra.
    """
    # Input validation
    if A.ndim == 1 or U.ndim == 1:
        raise ValueError(
            'Acceleration and shear matrices must contain vectors')

    if A.ndim != 2 or U.ndim != 2:
        raise ValueError(
            'Acceleration and shear matrices must be 2-dimensional')

    if A.shape[0] != U.shape[0]:
        raise ValueError(
            'Acceleration and shear matrices must have the same number of rows')

    if not isinstance(n_fft, int) or n_fft < 2:
        raise ValueError('n_fft must be an integer larger than 2.')

    if A.shape[0] < 2 * n_fft:
        raise ValueError(
            'Vector lengths must be at least 2 * n_fft samples long')

    if not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError('Sampling rate must be a positive scalar')

    # Spectral estimation parameters
    window = 'hann'
    n_overlap = n_fft // 2
    n_freqs = n_fft // 2 + 1

    # Number of accelerometers and shear probes
    n_samples, n_accel = A.shape
    _, n_shear = U.shape

    # Pre-allocate matrices for spectra
    AA = np.zeros((n_accel, n_accel, n_freqs), dtype=np.complex128)
    UU = np.zeros((n_shear, n_shear, n_freqs), dtype=np.complex128)
    UA = np.zeros((n_shear, n_accel, n_freqs), dtype=np.complex128)

    # Compute acceleration auto- and cross-spectra
    for i in range(n_accel):
        # Auto-spectrum for accelerometer i
        f, Paa = csd(A[:, i], A[:, i], fs=rate, nperseg=n_fft,
                     noverlap=n_overlap, window=window)
        AA[i, i, :] = Paa
        for j in range(i + 1, n_accel):
            # Cross-spectrum between accelerometers i and j
            _, Paa_cross = csd(A[:, i], A[:, j], fs=rate,
                               nperseg=n_fft, noverlap=n_overlap, window=window)
            AA[i, j, :] = Paa_cross
            AA[j, i, :] = np.conj(Paa_cross)

    # Compute shear probe auto- and cross-spectra
    for i in range(n_shear):
        # Auto-spectrum for shear probe i
        _, Puu = csd(U[:, i], U[:, i], fs=rate, nperseg=n_fft,
                     noverlap=n_overlap, window=window)
        UU[i, i, :] = Puu
        for j in range(i + 1, n_shear):
            # Cross-spectrum between shear probes i and j
            _, Puu_cross = csd(U[:, i], U[:, j], fs=rate,
                               nperseg=n_fft, noverlap=n_overlap, window=window)
            UU[i, j, :] = Puu_cross
            UU[j, i, :] = np.conj(Puu_cross)

    # Compute cross-spectra between shear probes and accelerometers
    for i in range(n_shear):
        for j in range(n_accel):
            # Cross-spectrum between shear probe i and accelerometer j
            _, Pua = csd(U[:, i], A[:, j], fs=rate, nperseg=n_fft,
                         noverlap=n_overlap, window=window)
            UA[i, j, :] = Pua

    # Initialize clean shear spectra
    clean_UU = np.zeros_like(UU, dtype=np.complex128)

    # Clean the shear spectra at each frequency
    for idx in range(len(f)):
        # Extract spectra at the current frequency
        UU_freq = UU[:, :, idx]
        UA_freq = UA[:, :, idx]
        AA_freq = AA[:, :, idx]

        # Invert AA_freq matrix (acceleration cross-spectra matrix)
        try:
            AA_inv = np.linalg.inv(AA_freq)
            # Compute the term to subtract
            term = UA_freq @ AA_inv @ UA_freq.conj().T
            # Cleaned shear spectra at this frequency
            clean_UU[:, :, idx] = UU_freq - term
        except np.linalg.LinAlgError:
            # If AA is singular, skip cleaning at this frequency
            clean_UU[:, :, idx] = UU_freq

    # Remove any singleton dimensions and take the real part
    clean_UU = np.real(np.squeeze(clean_UU))
    UU = np.real(np.squeeze(UU))
    AA = np.real(np.squeeze(AA))
    UA = np.squeeze(UA)

    return clean_UU, AA, UU, UA, f
