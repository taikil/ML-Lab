import numpy as np
from scipy.signal import csd


def clean(A, U, n_fft, rate):
    """
    Remove acceleration contamination from all shear channels.

    Parameters:
    A : ndarray
        Shape (n_samples, n_accel). Acceleration signals.
    U : ndarray
        Shape (n_samples, n_shear). Shear probe signals.
    n_fft : int
        FFT length for spectral estimation.
    rate : float
        Sampling rate in Hz.

    Returns:
    clean_UU : ndarray
        Cleaned shear cross-spectra, real-valued, shape (n_shear, n_shear, n_freqs).
    AA : ndarray
        Acceleration cross-spectra, shape (n_accel, n_accel, n_freqs).
    UU : ndarray
        Original shear cross-spectra before removal, shape (n_shear, n_shear, n_freqs).
    UA : ndarray
        Cross-spectra between shear and acceleration, shape (n_shear, n_accel, n_freqs).
    f : ndarray
        Frequency vector.
    """
    # Input validation
    if A.ndim == 1 or U.ndim == 1:
        raise ValueError(
            'Acceleration and shear must be matrices with 2D shape.')
    if A.shape[0] != U.shape[0]:
        raise ValueError('A and U must have the same number of rows.')
    if not isinstance(n_fft, int) or n_fft < 2:
        raise ValueError('n_fft must be an integer larger than 2.')
    if A.shape[0] < 2 * n_fft:
        raise ValueError('Signal must be longer than twice n_fft.')
    if not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError('Sampling rate must be positive.')

    np.savetxt("U_sh.txt", U, fmt='%.6e')

    # Create the same cosine window as in MATLAB
    # Window = 1 + cos(pi * (-1 + 2*(0:n_fft-1)/n_fft))
    # Normalize to have mean-square = 1
    w = 1 + np.cos(np.pi * (-1 + 2*np.arange(n_fft)/n_fft))
    w = w / np.sqrt(np.mean(w**2))

    # Overlap
    n_overlap = n_fft // 2

    # Dimensions
    n_samples, n_accel = A.shape
    _, n_shear = U.shape

    # Initialize spectra arrays
    # The number of frequency bins returned by csd is (n_fft//2 + 1)
    freqs_count = n_fft // 2 + 1
    AA = np.zeros((n_accel, n_accel, freqs_count), dtype=np.complex128)
    UU = np.zeros((n_shear, n_shear, freqs_count), dtype=np.complex128)
    UA = np.zeros((n_shear, n_accel, freqs_count), dtype=np.complex128)

    # Use scaling='spectrum' to get variance-preserving scaling
    # detrend=False matches the 'none' detrend in MATLAB
    scaling = 'spectrum'
    detrend = False

    # Compute acceleration auto- and cross-spectra
    for i in range(n_accel):
        f, Paa = csd(A[:, i], A[:, i], fs=rate, window=w,
                     nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
        AA[i, i, :] = Paa
        for j in range(i + 1, n_accel):
            _, Paa_cross = csd(A[:, i], A[:, j], fs=rate, window=w,
                               nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
            AA[i, j, :] = Paa_cross
            AA[j, i, :] = np.conj(Paa_cross)

    # Compute shear probe auto- and cross-spectra
    for i in range(n_shear):
        _, Puu = csd(U[:, i], U[:, i], fs=rate, window=w,
                     nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
        UU[i, i, :] = Puu
        for j in range(i + 1, n_shear):
            _, Puu_cross = csd(U[:, i], U[:, j], fs=rate, window=w,
                               nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
            UU[i, j, :] = Puu_cross
            UU[j, i, :] = np.conj(Puu_cross)

    # Compute cross-spectra between shear and acceleration
    for i in range(n_shear):
        for j in range(n_accel):
            _, Pua = csd(U[:, i], A[:, j], fs=rate, window=w,
                         nperseg=n_fft, noverlap=n_overlap, scaling=scaling, detrend=detrend)
            UA[i, j, :] = Pua

    # Now remove coherent acceleration-induced contamination
    clean_UU = np.zeros_like(UU, dtype=np.complex128)
    for idx in range(len(f)):
        UU_freq = UU[:, :, idx]
        UA_freq = UA[:, :, idx]
        AA_freq = AA[:, :, idx]

        # Use pseudoinverse in case AA is singular
        AA_inv = np.linalg.pinv(AA_freq)
        term = UA_freq @ AA_inv @ UA_freq.conj().T
        clean_UU[:, :, idx] = UU_freq - term

    # Convert cleaned spectra to real, as MATLAB does 'clean_UU=real(clean_UU);'
    clean_UU = clean_UU.real

    return clean_UU, AA, UU, UA, f
