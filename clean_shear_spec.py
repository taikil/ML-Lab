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

    window = np.hanning(n_fft)
    n_overlap = n_fft // 2
    scaling = 'density'
    detrend = False

    # Number of accelerometers and shear probes
    n_samples, n_accel = A.shape
    _, n_shear = U.shape

    # Pre-allocate matrices for spectra
    AA = np.zeros((n_accel, n_accel, n_fft // 2 + 1), dtype=np.complex128)
    UU = np.zeros((n_shear, n_shear, n_fft // 2 + 1), dtype=np.complex128)
    UA = np.zeros((n_shear, n_accel, n_fft // 2 + 1), dtype=np.complex128)

    # Compute acceleration auto- and cross-spectra
    for i in range(n_accel):
        f, Paa = csd(A[:, i], A[:, i], fs=rate, window=window,
                     nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
        AA[i, i, :] = Paa
        for j in range(i + 1, n_accel):
            _, Paa_cross = csd(A[:, i], A[:, j], fs=rate, window=window,
                               nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
            AA[i, j, :] = Paa_cross
            AA[j, i, :] = np.conj(Paa_cross)

    # Compute shear probe auto- and cross-spectra
    for i in range(n_shear):
        _, Puu = csd(U[:, i], U[:, i], fs=rate, window=window,
                     nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
        UU[i, i, :] = Puu
        for j in range(i + 1, n_shear):
            _, Puu_cross = csd(U[:, i], U[:, j], fs=rate, window=window,
                               nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
            UU[i, j, :] = Puu_cross
            UU[j, i, :] = np.conj(Puu_cross)

    # Compute cross-spectra between shear probes and accelerometers
    for i in range(n_shear):
        for j in range(n_accel):
            _, Pua = csd(U[:, i], A[:, j], fs=rate, window=window,
                         nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
            UA[i, j, :] = Pua

    # Initialize clean shear spectra
    clean_UU = np.zeros_like(UU, dtype=np.complex128)

    # Clean the shear spectra at each frequency
    for idx in range(len(f)):
        UU_freq = UU[:, :, idx]
        UA_freq = UA[:, :, idx]
        AA_freq = AA[:, :, idx]

        # Use pseudoinverse to handle singular matrices
        AA_inv = np.linalg.pinv(AA_freq)
        term = UA_freq @ AA_inv @ UA_freq.conj().T
        clean_UU[:, :, idx] = UU_freq - term

    # Remove singleton dimensions
    clean_UU = np.squeeze(clean_UU)
    clean_UU = np.real(clean_UU)
    UU = np.squeeze(UU)
    n_shear, _, n_freqs = UU.shape
    with open('UU_output.txt', 'w') as file:
        for freq_idx in range(n_freqs):
            file.write(f"Frequency Index {freq_idx}:\n")
            for i in range(n_shear):
                row = ''
                for j in range(n_shear):
                    val = UU[i, j, freq_idx]
                    # Format complex number
                    val_str = f"{val.real:+.6e} {val.imag:+.6e}i"
                    row += val_str + '\t'
                file.write(row.strip() + '\n')
            file.write('\n')
    AA = np.squeeze(AA)
    UA = np.squeeze(UA)

    return clean_UU, AA, UU, UA, f
