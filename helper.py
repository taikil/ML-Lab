import numpy as np
from scipy.interpolate import interp1d
import gsw  # Gibbs SeaWater Oceanographic Package of TEOS-10
from scipy.signal import butter, filtfilt, medfilt


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
    # C_az = np.fft.ifft(C_az, axis=0) * n_fft
    C_az = np.fft.ifft(C_az, axis=0)

    # Calculate biased cross-covariance of dw_dx and az
    dw_dx_padded = np.vstack([dw_dx[:L - N, :], np.zeros((N, n_channels))])
    C_dw_dx = np.fft.fft(dw_dx_padded, axis=0)
    # C_az_dw_dx = np.fft.ifft(C_dw_dx * np.conj(c_az),
    #                          axis=0) * n_fft  # Scale to match MATLAB
    C_az_dw_dx = np.fft.ifft(C_dw_dx * np.conj(c_az),
                             axis=0)

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


def compute_density(JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast):
    """
    Compute the potential density sigma_theta at the fast sampling rate.
    """
    JAC_T_smooth = moving_average(JAC_T, 50)
    JAC_C_smooth = moving_average(JAC_C, 50)

    # Ensure conductivity is in mS/cm
    # If JAC_C_smooth is in S/m, convert it
    JAC_C_smooth = JAC_C_smooth * 10  # Uncomment if needed

    # Compute Practical Salinity (SP) from conductivity, temperature, and pressure
    SP = gsw.SP_from_C(JAC_C_smooth, JAC_T_smooth, P_slow)

    longitude = np.full_like(P_slow, 135.0)  # Longitude in degrees East
    latitude = np.full_like(P_slow, 30.0)    # Latitude in degrees North

    SA = gsw.SA_from_SP(SP, P_slow, longitude, latitude)
    CT = gsw.CT_from_t(SA, JAC_T_smooth, P_slow)
    sigma_theta = gsw.sigma0(SA, CT)

    # Interpolate to fast sampling rate
    time_slow = np.arange(len(P_slow)) / fs_slow
    time_fast = np.arange(len(P_fast)) / fs_fast

    interp_func = interp1d(time_slow, sigma_theta,
                           kind='linear', fill_value='extrapolate')
    sigma_theta_fast = interp_func(time_fast)

    return sigma_theta_fast


def compute_buoyancy_frequency(sigma_theta_fast, P_fast):
    """
    Compute the squared buoyancy frequency N2.
    """
    window_size = 200  # Adjust as needed
    P_fast_smooth = moving_average(P_fast, window_size)
    sigma_theta_smooth = moving_average(sigma_theta_fast, window_size)
    sigma_theta_sorted = np.sort(sigma_theta_smooth)
    g = 9.81  # gravitational acceleration
    buoyi = -g * sigma_theta_sorted / 1025.0

    # Compute gradients
    db = np.gradient(buoyi)
    dz = np.gradient(P_fast_smooth)

    N2 = -db / dz
    return N2


def moving_average(data, window_size):
    """
    Compute the moving average of the data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def despike_and_filter_sh(sh1, sh2, Ax, Ay, n, fs_fast, params):
    """
    Despike and apply high-pass filter to shear data.
    """
    N = 15  # Number of wiener filter weights

    # Apply Wiener filter to remove acceleration-coherent noise
    _, sh1_clean = wiener(sh1[n], Ax[n], N)
    _, sh1_clean = wiener(sh1_clean, Ay[n], N)
    _, sh2_clean = wiener(sh2[n], Ax[n], N)
    _, sh2_clean = wiener(sh2_clean, Ay[n], N)

    # High-pass filter
    HP_cut = params['HP_cut']
    b_hp, a_hp = butter(4, HP_cut / (fs_fast / 2), btype='high')
    padlen = min(20, len(sh1_clean) - 1)

    sh1_HP = filtfilt(b_hp, a_hp, sh1_clean, padlen=padlen)
    sh2_HP = filtfilt(b_hp, a_hp, sh2_clean, padlen=padlen)

    return sh1_HP, sh2_HP
