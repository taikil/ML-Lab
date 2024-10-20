import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar


def get_diss_odas(shear, A, fft_length, diss_length, overlap, fs_fast, fs_slow,
                  speed, T, P, t_fast=None, Data_fast=None, Data_slow=None,
                  fit_order=3, f_AA=98, fit_2_isr=1.5e-5, f_limit=np.inf, UVW=None):
    """
    Calculate a profile of dissipation rate from shear probe data.

    Parameters:
    - shear: ndarray, matrix of shear probe signals, one probe per column.
    - A: ndarray, matrix of acceleration signals, one accelerometer per column.
    - fft_length: int, length of the FFT segment (samples).
    - diss_length: int, length of shear data over which to estimate dissipation (samples).
    - overlap: int, overlap between successive dissipation estimates (samples).
    - fs_fast: float, sampling rate for shear and acceleration (Hz).
    - fs_slow: float, sampling rate for slow channels in Data_slow (Hz).
    - speed: ndarray or float, profiling speed used to derive wavenumber spectra.
    - T: ndarray, temperature vector used when calculating kinematic viscosity.
    - P: ndarray, pressure data.

    Optional Parameters:
    - t_fast: ndarray, time vector for fast channels.
    - Data_fast: ndarray, fast channels data to be averaged over each dissipation estimate.
    - Data_slow: ndarray, slow channels data to be averaged over each dissipation estimate.
    - fit_order: int, order of polynomial fit to shear spectra in log-log space.
    - f_AA: float, cut-off frequency of the anti-aliasing filter (Hz).
    - fit_2_isr: float, dissipation rate threshold for inertial subrange fitting (W/kg).
    - f_limit: float, maximum frequency to use when estimating dissipation (Hz).
    - UVW: ndarray, matrix where each column is a velocity component (U, V, W).

    Returns:
    - diss: dict, contains the dissipation estimates and related information.
    """

    # Initialize dissipation structure
    diss = {
        'e': [],          # Dissipation rate estimates
        'K': [],          # Wavenumbers
        'K_max': [],      # Maximum wavenumber used in each estimate
        'sh_clean': [],   # Cleaned shear spectra
        'sh': [],         # Raw shear spectra
        'AA': [],         # Acceleration spectra
        'UA': [],         # Shear-Acceleration cross-spectra
        'Nasmyth_spec': [],  # Nasmyth spectra
        'F': [],          # Frequencies
        'speed': [],      # Mean speed for each estimate
        'nu': [],         # Kinematic viscosity for each estimate
        'P': [],          # Pressure at each estimate
        'T': [],          # Temperature at each estimate
        't': [],          # Time at each estimate
        'Data_fast': [],  # Averaged fast data
        'Data_slow': [],  # Averaged slow data
        # Method used for each estimate (0: variance, 1: inertial subrange)
        'method': [],
        'DOF': [],        # Degrees of freedom for each estimate
        'MAD': [],        # Mean absolute deviation for each estimate
        'AOA': [],        # Angle of attack (if UVW is provided)
    }

    # Set default values and handle optional parameters
    f_AA_limit = 0.9
    f_AA = f_AA_limit * f_AA  # Adjust anti-aliasing frequency
    if f_limit < f_AA:
        f_AA = f_limit

    estimate_AOA = UVW is not None

    # Calculate the number of dissipation estimates
    if overlap >= diss_length:
        overlap = diss_length / 2
    if overlap < 0:
        overlap = 0
    num_of_estimates = 1 + \
        int((shear.shape[0] - diss_length) / (diss_length - overlap))

    # Initialize arrays to store results
    num_of_shear = shear.shape[1]
    F_length = 1 + fft_length // 2  # Frequency vector length

    diss['e'] = np.zeros((num_of_shear, num_of_estimates))
    diss['K_max'] = np.zeros((num_of_shear, num_of_estimates))
    diss['method'] = np.zeros((num_of_shear, num_of_estimates))
    diss['MAD'] = np.zeros((num_of_shear, num_of_estimates))
    diss['DOF'] = np.zeros((num_of_shear, num_of_estimates))

    diss['Nasmyth_spec'] = np.zeros((F_length, num_of_shear, num_of_estimates))
    diss['sh_clean'] = np.zeros(
        (F_length, num_of_shear, num_of_shear, num_of_estimates), dtype=complex)
    diss['sh'] = np.zeros_like(diss['sh_clean'])
    diss['AA'] = np.zeros(
        (F_length, A.shape[1], A.shape[1], num_of_estimates), dtype=complex)
    diss['UA'] = np.zeros(
        (F_length, num_of_shear, A.shape[1], num_of_estimates), dtype=complex)
    diss['F'] = np.zeros((F_length, num_of_estimates))
    diss['K'] = np.zeros_like(diss['F'])
    diss['speed'] = np.zeros(num_of_estimates)
    diss['nu'] = np.zeros(num_of_estimates)
    diss['P'] = np.zeros(num_of_estimates)
    diss['T'] = np.zeros(num_of_estimates)
    diss['t'] = np.zeros(num_of_estimates)

    if estimate_AOA:
        diss['AOA'] = np.zeros(num_of_estimates)

    # Main loop over dissipation estimates
    index = 0
    select = np.arange(diss_length)

    while select[-1] < shear.shape[0]:
        # Select data segments
        shear_segment = shear[select, :]
        A_segment = A[select, :]

        # Compute cleaned shear spectrum and other spectra
        P_sh_clean, AA, P_sh, UA, F = clean_shear_spec(
            A_segment, shear_segment, fft_length, fs_fast)

        # Convert frequency spectra to wavenumber spectra
        W = np.mean(np.abs(speed[select]))
        K = F / W
        K_AA = f_AA / W  # Anti-aliasing wavenumber

        # Apply wavenumber correction
        correction = np.ones_like(K)
        correction_index = K <= 150
        correction[correction_index] = 1 + (K[correction_index] / 48) ** 2
        P_sh_clean *= W * correction[:, np.newaxis, np.newaxis]
        P_sh *= W * correction[:, np.newaxis, np.newaxis]

        # Initialize arrays for dissipation calculations
        e = np.zeros(num_of_shear)
        K_max = np.zeros(num_of_shear)
        method = np.zeros(num_of_shear)
        DOF = np.zeros(num_of_shear)
        MAD = np.zeros(num_of_shear)

        mean_T = np.mean(T[select])
        mean_P = np.mean(P[select])
        mean_t = np.mean(t_fast[select]) if t_fast is not None else 0
        nu = visc35(mean_T)  # Kinematic viscosity

        # Compute dissipation for each shear probe
        for i in range(num_of_shear):
            shear_spectrum = np.squeeze(P_sh_clean[:, i, i])

            # Estimate initial dissipation rate e_1
            K_range = K <= 10
            e_10 = 7.5 * nu * np.trapz(shear_spectrum[K_range], K[K_range])
            e_1 = e_10 * np.sqrt(1 + 1.0774e9 * e_10)  # Lueck's model

            if e_1 < fit_2_isr:
                # Variance method
                e_4, K_max_i = variance_method(
                    K, shear_spectrum, e_1, nu, K_AA, fit_order)
                method[i] = 0
            else:
                # Inertial subrange fitting method
                e_4, K_max_i, _ = inertial_subrange(
                    K, shear_spectrum, e_1, nu, K_AA)
                method[i] = 1

            e[i] = e_4
            K_max[i] = K_max_i
            DOF[i] = 2 * (9 / 11) * 2 * (diss_length //
                                         fft_length) * (len(K_range) - 1)
            Nasmyth_spec = nasmyth_spectrum(K, e_4, nu)
            diss['Nasmyth_spec'][:, i, index] = Nasmyth_spec
            MAD[i] = np.mean(
                np.abs(np.log10(shear_spectrum[K_range][1:] / Nasmyth_spec[K_range][1:])))

        # Store results in diss structure
        diss['e'][:, index] = e
        diss['K_max'][:, index] = K_max
        diss['method'][:, index] = method
        diss['DOF'][:, index] = DOF
        diss['MAD'][:, index] = MAD
        diss['sh_clean'][:, :, :, index] = P_sh_clean
        diss['sh'][:, :, :, index] = P_sh
        diss['AA'][:, :, :, index] = AA
        diss['UA'][:, :, :, index] = UA
        diss['F'][:, index] = F
        diss['K'][:, index] = K
        diss['speed'][index] = W
        diss['nu'][index] = nu
        diss['P'][index] = mean_P
        diss['T'][index] = mean_T
        diss['t'][index] = mean_t

        if estimate_AOA:
            U, V, W_velocity = UVW[select, 0], UVW[select, 1], UVW[select, 2]
            AOA = np.degrees(np.arctan2(
                np.sqrt(V**2 + W_velocity**2), np.abs(U))).max()
            diss['AOA'][index] = AOA

        index += 1
        select += diss_length - int(overlap)

    # Remove unused preallocated space if necessary
    # ...

    # Add processing parameters to diss structure
    diss['fs_fast'] = fs_fast
    diss['fs_slow'] = fs_slow
    diss['f_AA'] = f_AA
    diss['f_limit'] = f_limit
    diss['fit_order'] = fit_order
    diss['diss_length'] = diss_length
    diss['overlap'] = overlap
    diss['fft_length'] = fft_length

    return diss


def clean_shear_spec(A_segment, shear_segment, fft_length, fs_fast):
    """
    Compute cleaned shear spectrum and related spectra.

    Returns:
    - P_sh_clean: ndarray, cleaned shear spectra.
    - AA: ndarray, acceleration spectra.
    - P_sh: ndarray, raw shear spectra.
    - UA: ndarray, shear-acceleration cross-spectra.
    - F: ndarray, frequency vector.
    """
    # Number of segments for averaging
    nfft = fft_length
    noverlap = 0  # Adjust if needed
    window = signal.windows.hann(nfft)

    # Compute spectra using Welch's method
    F, P_sh = signal.welch(shear_segment, fs=fs_fast, window=window, nperseg=nfft,
                           noverlap=noverlap, axis=0, return_onesided=True)
    _, AA = signal.welch(A_segment, fs=fs_fast, window=window, nperseg=nfft,
                         noverlap=noverlap, axis=0, return_onesided=True)
    # Cross-spectra between shear and acceleration
    UA = np.zeros(
        (len(F), shear_segment.shape[1], A_segment.shape[1]), dtype=complex)
    for i in range(shear_segment.shape[1]):
        for j in range(A_segment.shape[1]):
            _, P_cross = signal.csd(shear_segment[:, i], A_segment[:, j], fs=fs_fast,
                                    window=window, nperseg=nfft, noverlap=noverlap,
                                    return_onesided=True)
            UA[:, i, j] = P_cross

    # Clean the shear spectra using Goodman coherent noise removal
    P_sh_clean = P_sh.copy()
    # Implement noise removal algorithm as needed
    # ...

    return P_sh_clean, AA, P_sh, UA, F


def nasmyth_spectrum(K, epsilon, nu):
    """
    Compute the Nasmyth spectrum.

    Parameters:
    - K: ndarray, wavenumber array (cpm).
    - epsilon: float, dissipation rate (W/kg).
    - nu: float, kinematic viscosity (m^2/s).

    Returns:
    - Phi: ndarray, theoretical shear spectrum values.
    """
    alpha = 1.5
    beta = 5.2
    eta = (nu**3 / epsilon)**0.25  # Kolmogorov length scale
    k_eta = K * eta
    Phi = (alpha * epsilon**0.5 * K**(-1)) / (1 + (beta * k_eta**(4)))**(5/3)
    return Phi


def inertial_subrange(K, shear_spectrum, e, nu, K_limit):
    """
    Fit the shear spectrum to the Nasmyth spectrum in the inertial subrange.

    Returns:
    - e: float, estimated dissipation rate.
    - K_max: float, maximum wavenumber used.
    - fit_range: ndarray, indices of K used in the fit.
    """
    x_isr = 0.02  # Adjusted nondimensional limit
    fit_range = K <= min([x_isr * e**0.25 / nu**0.75, K_limit])
    K_max = K[fit_range][-1]

    for _ in range(3):
        Nasmyth_values = nasmyth_spectrum(K[fit_range], e, nu)
        fit_error = np.mean(
            np.log10(shear_spectrum[fit_range][1:] / Nasmyth_values[1:]))
        e *= 10 ** (1.5 * fit_error)

    # Remove outliers
    Nasmyth_values = nasmyth_spectrum(K[fit_range], e, nu)
    fit_error = np.log10(shear_spectrum[fit_range][1:] / Nasmyth_values[1:])
    flyers_index = np.abs(fit_error) > 0.5
    if np.any(flyers_index):
        max_remove = int(0.2 * len(fit_range))
        remove_indices = np.argsort(-np.abs(fit_error))[:max_remove]
        fit_range = np.delete(fit_range, remove_indices)
        shear_spectrum = shear_spectrum[fit_range]
    K_max = K[fit_range][-1]

    # Refit with reduced spectrum
    for _ in range(2):
        Nasmyth_values = nasmyth_spectrum(K[fit_range], e, nu)
        fit_error = np.mean(
            np.log10(shear_spectrum[fit_range][1:] / Nasmyth_values[1:]))
        e *= 10 ** (1.5 * fit_error)

    return e, K_max, fit_range


def variance_method(K, shear_spectrum, e_1, nu, K_AA, fit_order):
    """
    Estimate dissipation rate using the variance method.

    Returns:
    - e_new: float, estimated dissipation rate.
    - K_max: float, maximum wavenumber used.
    """
    x_95 = 0.1205
    K_95 = x_95 * (e_1 / nu**3) ** 0.25
    K_limit = min([np.log10(K_AA), np.log10(K_95), np.log10(150)])
    K_limit = np.clip(K_limit, np.log10(10), np.log10(150))
    Range = K <= 10 ** K_limit
    e_3 = 7.5 * nu * np.trapz(shear_spectrum[Range], K[Range])

    x_limit = (K[Range][-1] * (nu**3 / e_3) ** 0.25) ** (4/3)
    variance_resolved = np.tanh(48 * x_limit) - \
        2.9 * x_limit * np.exp(-22.3 * x_limit)
    e_new = e_3 / variance_resolved

    # Iteratively adjust e_new
    for _ in range(5):
        x_limit = (K[Range][-1] * (nu**3 / e_new) ** 0.25) ** (4/3)
        variance_resolved = np.tanh(
            48 * x_limit) - 2.9 * x_limit * np.exp(-22.3 * x_limit)
        e_old = e_new
        e_new = e_3 / variance_resolved
        if np.abs(e_new / e_old - 1) < 0.02:
            break

    K_max = K[Range][-1]
    return e_new, K_max


def visc35(T):
    """
    Calculate kinematic viscosity based on temperature (simplified).

    Parameters:
    - T: float, temperature (°C).

    Returns:
    - nu: float, kinematic viscosity (m^2/s).
    """
    # Simplified empirical formula for seawater at 35 PSU
    nu = 1e-6 * (17.91 - 0.5381 * T + 0.00694 * T**2)
    return nu
