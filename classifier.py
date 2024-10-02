import sys
import numpy as np
import scipy.io
import h5py
import scipy.signal as signal
from diss_rate_odas_nagai import *
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
import gsw  # Gibbs SeaWater Oceanographic Package of TEOS-10
import matplotlib.pyplot as plt
import hdf5storage


FILENAME = ''


def get_file():
    if len(sys.argv) == 2:
        filename = str(sys.argv[1])
        return filename
    else:
        print("Invalid Input; Provide the .mat filename as a command-line argument.")
        print("Exiting Program...")
        sys.exit()


def load_mat_file(filename):
    try:
        mat_contents = scipy.io.loadmat(
            filename, struct_as_record=False, squeeze_me=True)
        data = mat_contents.get('data', None)
        dataset = mat_contents.get('dataset', None)

        if data is None or dataset is None:
            raise ValueError(
                "The .mat file does not contain both 'data' and 'dataset' variables.")

        print("Loaded .mat file using scipy.io.loadmat.")
        return data, dataset
    except NotImplementedError:
        # Fallback to hdf5storage for MATLAB v7.3 files
        print("Loading .mat file using hdf5storage.")
        mat_contents = hdf5storage.loadmat(filename)
        data = mat_contents.get('data', None)
        dataset = mat_contents.get('dataset', None)
        if data is None or dataset is None:
            raise ValueError(
                "The .mat file does not contain both 'data' and 'dataset' variables.")
        return data, dataset


def process_profile(data, dataset, params, profile_num=0):
    """
    Process the profile data to calculate the dissipation rate.
    """
    # Extract sampling frequencies from 'data'
    fs_fast = np.squeeze(data['fs_fast'])
    fs_slow = np.squeeze(data['fs_slow'])

    fs_fast = float(fs_fast)
    fs_slow = float(fs_slow)

    # Get the number of profiles
    num_profiles = dataset['P_slow'].shape[1]
    if profile_num >= num_profiles:
        raise ValueError(
            f"Profile number {profile_num} exceeds available profiles ({num_profiles}).")

    # Extract variables for the selected profile
    try:
        P_slow = np.squeeze(dataset['P_slow'][0, profile_num])
        JAC_T = np.squeeze(dataset['JAC_T'][0, profile_num])
        JAC_C = np.squeeze(dataset['JAC_C'][0, profile_num])
        P_fast = np.squeeze(dataset['P_fast'][0, profile_num])
        W_slow = np.squeeze(dataset['W_slow'][0, profile_num])
        W_fast = np.squeeze(dataset['W_fast'][0, profile_num])
        sh1 = np.squeeze(dataset['sh1'][0, profile_num])
        sh2 = np.squeeze(dataset['sh2'][0, profile_num])
        Ax = np.squeeze(dataset['Ax'][0, profile_num])
        Ay = np.squeeze(dataset['Ay'][0, profile_num])
        T1_fast = np.squeeze(dataset['T1_fast'][0, profile_num])
    except KeyError as e:
        raise KeyError(f"Missing expected field in dataset: {e}")

    # Compute derived quantities
    sigma_theta_fast = compute_density(
        JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast)
    N2 = compute_buoyancy_frequency(sigma_theta_fast, P_fast)

    # Prepare profile indices
    n, m = get_profile_indices(P_slow, W_slow, params, fs_slow, fs_fast)

    # Despike and filter shear data
    sh1_HP, sh2_HP = despike_and_filter_sh(
        sh1, sh2, Ax, Ay, n, fs_fast, params)

    # Prepare data for dissipation rate calculation
    diss = calculate_dissipation_rate(
        sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, P_fast, N2, n, params, fs_fast
    )

    return diss


def compute_density(JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast):
    """
    Compute the potential density sigma_theta at the fast sampling rate.
    """
    JAC_T_smooth = moving_average(JAC_T, 50)
    JAC_C_smooth = moving_average(JAC_C, 50)

    # Ensure conductivity is in mS/cm
    # If JAC_C_smooth is in S/m, convert it
    # JAC_C_smooth = JAC_C_smooth * 10  # Uncomment if needed

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
    # Smooth pressure and density
    window_size = 200  # Adjust as needed
    P_fast_smooth = moving_average(P_fast, window_size)
    sigma_theta_smooth = moving_average(sigma_theta_fast, window_size)

    # Compute buoyancy frequency squared
    g = 9.81  # gravitational acceleration
    db = np.gradient(-g * sigma_theta_smooth / 1025.0)
    dz = np.gradient(P_fast_smooth)
    N2 = -db / dz
    return N2


def moving_average(data, window_size):
    """
    Compute the moving average of the data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def get_profile_indices(P_slow, W_slow, params, fs_slow, fs_fast):
    """
    Determine the start and end indices for the profile.
    """
    # Implement logic similar to 'get_profile' in MATLAB code
    # For simplicity, we'll select the entire range
    start_index_slow = 0
    end_index_slow = len(P_slow) - 1
    start_index_fast = int((fs_fast / fs_slow) * start_index_slow)
    end_index_fast = int((fs_fast / fs_slow) * end_index_slow)
    n = np.arange(start_index_fast, end_index_fast + 1)
    m = np.arange(start_index_slow, end_index_slow + 1)
    return n, m


def despike_and_filter_sh(sh1, sh2, Ax, Ay, n, fs_fast, params):
    """
    Despike and apply high-pass filter to shear data.
    """
    N = 15  # Number of Wiener filter weights

    # Apply Wiener filter to remove acceleration-coherent noise
    _, sh1_clean = wiener(sh1[n], Ax[n], N)
    _, sh1_clean = wiener(sh1_clean, Ay[n], N)
    _, sh2_clean = wiener(sh2[n], Ax[n], N)
    _, sh2_clean = wiener(sh2_clean, Ay[n], N)

    # High-pass filter
    HP_cut = params['HP_cut']
    b_hp, a_hp = butter(4, HP_cut / (fs_fast / 2), btype='high')
    sh1_HP = filtfilt(b_hp, a_hp, sh1_clean)
    sh2_HP = filtfilt(b_hp, a_hp, sh2_clean)

    return sh1_HP, sh2_HP


def calculate_dissipation_rate(
    sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, P_fast, N2, n, params, fs_fast
):
    """
    Calculate the dissipation rate using the processed data.
    """
    # Select pressure range
    P_start = params['P_start']
    P_end = params['P_end']
    range_indices = n[(P_fast[n] >= P_start) & (P_fast[n] <= P_end)]

    if len(range_indices) == 0:
        raise ValueError(
            "No data points found in the specified pressure range.")

    # Prepare variables
    sh = np.column_stack((sh1_HP[range_indices], sh2_HP[range_indices]))
    A = np.column_stack((Ax[range_indices], Ay[range_indices]))
    press = P_fast[range_indices]
    W = W_fast[range_indices]
    T = T1_fast[range_indices]
    nu = visc35(T)
    N2_range = N2[range_indices]

    # Prepare info for dissipation calculation
    info = {
        'fft_length': int(params['fft_length'] * fs_fast),
        'diss_length': int(params['diss_length'] * fs_fast),
        'overlap': int(params['overlap'] * fs_fast),
        'fs': fs_fast,
        'speed': W,
        'T': T,
        'P': press,
        'N2': N2_range,
        'fit_order': 5,
        'f_AA': 98,
        'fit_2_Nasmyth': params['fit_2_Nasmyth']
    }
    handles = {
        'axes1': plt.subplot(1, 3, 1),
        'axes2': plt.subplot(1, 3, 2),
        'axes3': plt.subplot(1, 3, 3),
        'text14': None  # If you have any text labels to update
    }

    # Calculate dissipation rate
    diss = get_diss_odas_nagai4gui2024(
        SH=sh,
        A=A,
        fft_length=info['fft_length'],
        diss_length=info['diss_length'],
        overlap=info['overlap'],
        fs=fs_fast,
        speed=W,
        T=T,
        N2=N2_range,
        P=press,
        fit_order=info.get('fit_order', 5),
        f_AA=info.get('f_AA', 98),
    )

    # Extract data for plotting
    K = diss['K']               # Wavenumber array
    P_sh = diss['sh_clean']         # Measured shear spectra
    epsilon = diss['e']   # Dissipation rate estimates
    P_nas = diss['Nasmyth_spec']       # Nasmyth spectra

    selected_row = 0  # You can change this to plot different rows

    num_probes = P_sh.shape[1]

    for probe_index in range(num_probes):
        K_row = K[selected_row, :]  # Shape: (F_length,)
        # Extract auto-spectrum for the probe (auto-spectrum is when probe_i == probe_j)
        P_sh_probe = P_sh[selected_row, probe_index,
                          probe_index, :]  # Shape: (F_length,)
        P_nas_probe = P_nas[selected_row, probe_index, :]  # Shape: (F_length,)

        # Handle potential zeros in P_nas_probe to avoid division by zero
        P_nas_probe_safe = np.where(P_nas_probe == 0, np.nan, P_nas_probe)

        # Detect divergence point
        divergence_index = detect_divergence_point_threshold(
            K_row, P_sh_probe, P_nas_probe_safe, R_threshold=2.0
        )
        K_div = K_row[divergence_index]

        plt.figure(figsize=(8, 6))
        plt.loglog(K_row, P_sh_probe, label='Measured Spectrum')
        plt.loglog(K_row, P_nas_probe, label='Nasmyth Spectrum')
        plt.axvline(K_div, color='r', linestyle='--', label='Divergence Point')
        plt.xlabel('Wavenumber [cpm]')
        plt.ylabel('Shear Spectrum [(s$^{-1}$)$^2$/cpm]')
        plt.title(
            f'Shear Spectrum - Probe {probe_index + 1} at Row {selected_row}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

        return diss


def save_dissipation_rate(diss, profile_num):
    """
    Save the dissipation rate data to a .mat file.
    """
    filename = f'dissrate_profile_{profile_num}.mat'
    scipy.io.savemat(filename, {'diss': diss})
    print(f"Dissipation rate saved to {filename}")


def detect_divergence_point_threshold(K, P_measured, P_Nasmyth, R_threshold=2.0):
    """
    Detect divergence point based on a threshold ratio between measured and Nasmyth spectra.

    Parameters:
    - K: array of wavenumbers
    - P_measured: array of measured shear spectrum values
    - P_Nasmyth: array of Nasmyth spectrum values
    - R_threshold: threshold ratio to determine divergence

    Returns:
    - divergence_index: index of divergence point in K
    """
    ratio = P_measured / P_Nasmyth
    divergence_indices = np.where(ratio > R_threshold)[0]
    if len(divergence_indices) > 0:
        divergence_index = divergence_indices[0]
    else:
        divergence_index = len(K) - 1  # No divergence detected within range
    return divergence_index


def main():
    params = {
        'HP_cut': 1.0,          # High-pass filter cutoff frequency (Hz)
        'LP_cut': 10.0,         # Low-pass filter cutoff frequency (Hz)
        'P_min': 5.0,           # Minimum pressure (dbar)
        'W_min': 0.1,           # Minimum profiling speed (m/s)
        'direction': 'down',    # Profiling direction
        'fft_length': 4.0,      # FFT length (s)
        'diss_length': 8.0,     # Dissipation length (s)
        'overlap': 4.0,         # Overlap length (s)
        'fit_2_Nasmyth': 0,     # Fit to Nasmyth spectrum (boolean)
        'min_duration': 60.0,   # Minimum profile duration (s)
        # Profile number to process (fixed to 1 for naming)
        'profile_num': 1,
        'P_start': 0.0,         # Start pressure (dbar)
        'P_end': 1000.0         # End pressure (dbar)
    }
    FILENAME = get_file()
    data, dataset = load_mat_file(FILENAME)

    if isinstance(dataset, (np.ndarray, list)):
        num_profiles = dataset['P_slow'].shape[1]
        print(f"Number of profiles in dataset: {num_profiles}")
        # Loop over all profiles or prompt for specific profile
        profile_num = int(
            input(f"Enter profile number to process (0 to {num_profiles - 1}): "))
        diss = process_profile(data, dataset, params, profile_num)
        save_dissipation_rate(diss, profile_num)
    else:
        # Process single profile
        diss = process_profile(data, dataset, params)
        save_dissipation_rate(diss, params.get('profile_num', 1))

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
