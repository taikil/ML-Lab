import numpy as np


def wiener(dw_dx, az, N):
    """
    Implement Wiener filter to remove acceleration-coherent noise from a shear signal.

    Parameters:
    - dw_dx (array_like): Contaminated shear signal (shape: [L] or [L, n_channels]).
    - az (array_like): Acceleration signal causing contamination (same shape as dw_dx).
    - N (int): Desired number of filter weights.

    Returns:
    - w (ndarray): Wiener filter coefficients (shape: [N+1, n_channels]).
    - dw_dx_clean (ndarray): Cleaned shear signal (same shape as dw_dx).
    """
    import numpy as np
    from scipy.linalg import toeplitz
    from scipy.signal import lfilter

    # Ensure inputs are numpy arrays
    dw_dx = np.atleast_2d(dw_dx)
    az = np.atleast_2d(az)

    # Transpose if inputs are row vectors
    if dw_dx.shape[0] == 1:
        dw_dx = dw_dx.T
    if az.shape[0] == 1:
        az = az.T

    # Check that inputs have the same shape
    if dw_dx.shape != az.shape:
        raise ValueError('Both inputs must have the same size')

    L = az.shape[0]
    n_channels = az.shape[1]

    # Calculate biased autocovariance of az
    az_padded = np.vstack([az[:L - N, :], np.zeros((N, n_channels))])
    n_fft = az_padded.shape[0]
    c_az = np.fft.fft(az_padded, axis=0)
    C_az = np.abs(c_az) ** 2
    # Scale to match MATLAB's unscaled IFFT
    C_az = np.fft.ifft(C_az, axis=0) * n_fft

    # Calculate biased cross-covariance of dw_dx and az
    dw_dx_padded = np.vstack([dw_dx[:L - N, :], np.zeros((N, n_channels))])
    C_dw_dx = np.fft.fft(dw_dx_padded, axis=0)
    C_az_dw_dx = np.fft.ifft(C_dw_dx * np.conj(c_az),
                             axis=0) * n_fft  # Scale to match MATLAB

    # Keep only the first N+1 lags
    C_az = np.real(C_az[:N + 1, :])
    C_az_dw_dx = np.real(C_az_dw_dx[:N + 1, :])

    # Initialize output arrays
    w = np.zeros((N + 1, n_channels))
    dw_dx_clean = np.zeros_like(dw_dx)

    for k in range(n_channels):
        # Form Toeplitz matrix for autocovariance
        R = toeplitz(C_az[:, k])
        # Solve for Wiener filter coefficients
        w[:, k] = np.linalg.solve(R, C_az_dw_dx[:, k])
        # Estimate the contamination
        contamination_estimate = lfilter(w[:, k], [1], az[:, k])
        # Subtract estimated contamination to get the cleaned signal
        dw_dx_clean[:, k] = dw_dx[:, k] - contamination_estimate

    # If the original signals were 1D, return 1D arrays
    if dw_dx_clean.shape[1] == 1:
        dw_dx_clean = dw_dx_clean.ravel()
        w = w.ravel()

    return w, dw_dx_clean
