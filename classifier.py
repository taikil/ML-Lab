import sys
import numpy as np
import scipy.io
import h5py
import scipy.signal as signal
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
import gsw  # Gibbs SeaWater Oceanographic Package of TEOS-10
import matplotlib.pyplot as plt
from diss_rate_odas_nagai import *


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
        # Attempt to load using scipy.io.loadmat
        mat_contents = scipy.io.loadmat(
            filename, struct_as_record=False, squeeze_me=True)
        data = mat_contents.get('data', None)
        dataset = mat_contents.get('dataset', None)

        if data is None or dataset is None:
            raise ValueError(
                "The .mat file does not contain 'data' and 'dataset' variables.")

        print("Loaded .mat file using scipy.io.loadmat (MATLAB v7.2 or below).")
        return data, dataset
    except NotImplementedError:
        # Fallback to h5py for MATLAB v7.3
        print("Loading .mat file using h5py (MATLAB v7.3).")
        with h5py.File(filename, 'r') as f:
            # Extract 'data' and 'dataset' assuming they are top-level groups
            if 'data' in f.keys() and 'dataset' in f.keys():
                data = extract_h5py_group(f['data'], f)
                dataset = extract_h5py_group(f['dataset'], f)
            else:
                raise ValueError(
                    "The .mat file does not contain 'data' and 'dataset' variables.")

            # Additional debugging: Print structure of 'data' and 'dataset'
            print("Structure of 'data':")
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    print(f"  - {key}: {data[key].shape}")
                else:
                    print(f"  - {key}: {type(data[key])}")

            print("Structure of 'dataset':")
            for key in dataset.keys():
                if isinstance(dataset[key], np.ndarray):
                    print(f"  - {key}: {dataset[key].shape}")
                else:
                    print(f"  - {key}: {type(dataset[key])}")

            return data, dataset


def extract_h5py_group(group, file_handle):
    """
    Recursively extract data from an h5py group into a dictionary.
    """
    out = {}
    for key in group.keys():
        item = group[key]
        # Handle groups recursively
        if isinstance(item, h5py.Group):
            out[key] = extract_h5py_group(item, file_handle)
        # Handle datasets
        elif isinstance(item, h5py.Dataset):
            data = item[()]
            # Check if data is a reference
            if isinstance(data, h5py.Reference):
                data = file_handle[data][()]
            # Check if data is an array of references
            elif isinstance(data, np.ndarray) and data.dtype == object:
                data = np.array([file_handle[ref][()] if isinstance(
                    ref, h5py.Reference) else ref for ref in data])
            # Convert byte strings to regular strings if necessary
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            out[key] = data
        else:
            out[key] = item
    return out


def process_profile(data, dataset, params, model=None):
    """
    Process the profile data to calculate the dissipation rate.
    """
    # Use 'data' as the profile data
    profile = data

    # Extract and squeeze variables with error handling
    try:
        P_slow = np.squeeze(profile['P_slow'])
        JAC_T = np.squeeze(profile['JAC_T'])
        JAC_C = np.squeeze(profile['JAC_C'])
        P_fast = np.squeeze(profile['P_fast'])
        W_slow = np.squeeze(profile['W_slow'])
        W_fast = np.squeeze(profile['W_fast'])
        sh1 = np.squeeze(profile['sh1'])
        sh2 = np.squeeze(profile['sh2'])
        Ax = np.squeeze(profile['Ax'])
        Ay = np.squeeze(profile['Ay'])
        T1_fast = np.squeeze(profile['T1_fast'])
    except KeyError as e:
        raise KeyError(f"Missing expected field in profile: {e}")

    print("Shapes of extracted variables after squeezing:")
    print(f"P_slow: {P_slow.shape}")
    print(f"JAC_T: {JAC_T.shape}")
    print(f"JAC_C: {JAC_C.shape}")
    print(f"P_fast: {P_fast.shape}")
    print(f"W_slow: {W_slow.shape}")
    print(f"W_fast: {W_fast.shape}")
    print(f"sh1: {sh1.shape}")
    print(f"sh2: {sh2.shape}")
    print(f"Ax: {Ax.shape}")
    print(f"Ay: {Ay.shape}")
    print(f"T1_fast: {T1_fast.shape}")

    # Extract scalar sampling frequencies
    fs_fast = profile['fs_fast'].item()  # Convert from array to scalar
    fs_slow = profile['fs_slow'].item()  # Convert from array to scalar

    # Compute derived quantities
    sigma_theta_fast = compute_density(
        JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast)
    N2 = compute_buoyancy_frequency(sigma_theta_fast, P_fast)

    # Prepare profile indices
    n, m = get_profile_indices(P_slow, W_slow, params, fs_slow, fs_fast)

    # Despike and filter shear data
    sh1_HP, sh2_HP = despike_and_filter_sh(sh1, sh2, n, fs_fast, params)

    # Prepare data for dissipation rate calculation
    diss = calculate_dissipation_rate(
        sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, P_fast, N2, n, params, fs_fast
    )

    return diss


def compute_density(JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast):
    """
    Compute the potential density sigma_theta at the fast sampling rate.
    """
    # Smooth temperature and conductivity (if needed)
    JAC_T_smooth = moving_average(JAC_T, 50)
    JAC_C_smooth = moving_average(JAC_C, 50)

    # Convert practical salinity to Absolute Salinity
    # Assuming longitude and latitude are zero
    SA = gsw.SA_from_SP(JAC_C_smooth, P_slow, 0, 0)
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


def despike_and_filter_sh(sh1, sh2, n, fs_fast, params):
    """
    Despike and apply high-pass filter to shear data.
    """
    # Despike using median filter
    sh1_despiked = medfilt(sh1[n], kernel_size=7)
    sh2_despiked = medfilt(sh2[n], kernel_size=7)

    # High-pass filter
    HP_cut = params['HP_cut']
    b_hp, a_hp = butter(4, HP_cut / (fs_fast / 2), btype='high')
    sh1_HP = filtfilt(b_hp, a_hp, sh1_despiked)
    sh2_HP = filtfilt(b_hp, a_hp, sh2_despiked)

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
    P_sh = diss['P_sh']         # Measured shear spectra
    epsilon = diss['epsilon']   # Dissipation rate estimates
    P_nas = diss['P_nas']       # Nasmyth spectra

    # Plot the results
    for i in range(P_sh.shape[0]):  # Loop over probes
        divergence_index = detect_divergence_point_threshold(
            K, P_sh[i], P_nas[i], R_threshold=2.0
        )
        K_div = K[divergence_index]

        plt.figure(figsize=(8, 6))
        plt.loglog(K, P_sh[i], label='Measured Spectrum')
        plt.loglog(K, P_nas[i], label='Nasmyth Spectrum')
        plt.axvline(K_div, color='r', linestyle='--', label='Divergence Point')
        plt.xlabel('Wavenumber [cpm]')
        plt.ylabel('Shear Spectrum [(s$^{-1}$)$^2$/cpm]')
        plt.title(f'Shear Spectrum - Probe {i+1}')
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

    diss = process_profile(data, dataset, params)

    # Save the results with profile_num set to 1
    save_dissipation_rate(diss, params['profile_num'])

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
