import sys
import numpy as np
import scipy.io
from diss_rate_odas_nagai import *
from helper import *
from scipy.signal import butter, filtfilt, medfilt
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
    # Wavenumber array, shape: (num_segments, F_length)
    K = diss['K']
    # Measured shear spectra, shape: (num_segments, num_probes, num_probes, F_length)
    P_sh = diss['sh_clean']
    # Dissipation rate estimates, shape: (num_segments, num_probes)
    epsilon = diss['e']
    # Nasmyth spectra, shape: (num_segments, num_probes, F_length)
    P_nas = diss['Nasmyth_spec']
    num_segments = diss['e'].shape[0]
    num_probes = P_sh.shape[1]

    for segment_index in range(num_segments):
        for probe_index in range(num_probes):
            K_row = K[segment_index, :]  # Shape: (F_length,)
            # Extract auto-spectrum for the probe (auto-spectrum is when probe_i == probe_j)
            P_sh_probe = P_sh[segment_index, probe_index,
                              probe_index, :]  # Shape: (F_length,)
            P_nas_probe = P_nas[segment_index,
                                probe_index, :]  # Shape: (F_length,)
            e_segment_probe = epsilon[segment_index, probe_index]

            # Handle potential zeros in P_nas_probe to avoid division by zero
            P_nas_probe_safe = np.where(P_nas_probe == 0, np.nan, P_nas_probe)

            # Detect divergence point
            divergence_index = detect_divergence_point_threshold(
                K_row, P_sh_probe, P_nas_probe_safe, R_threshold=2.0
            )
            K_div = K_row[divergence_index]

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.loglog(K_row, P_sh_probe, label='Measured Spectrum')
            plt.loglog(K_row, P_nas_probe, label='Nasmyth Spectrum')
            plt.axvline(K_div, color='r', linestyle='--',
                        label='Divergence Point')
            plt.xlabel('Wavenumber [cpm]')
            plt.ylabel('Shear Spectrum [(s$^{-1}$)$^2$/cpm]')
            plt.title(
                f'Segment {segment_index + 1}, Probe {probe_index + 1}\nDissipation Rate: {e_segment_probe:.2e} W/kg')
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
