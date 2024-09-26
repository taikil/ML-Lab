import sys
import numpy as np
import scipy.io
import scipy.signal as signal
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
import gsw  # Gibbs SeaWater Oceanographic Package of TEOS-10
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
    """
    Load the .mat file and extract 'data' and 'dataset' variables.
    """
    mat_contents = scipy.io.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    data = mat_contents.get('data', None)
    dataset = mat_contents.get('dataset', None)

    if data is None or dataset is None:
        raise ValueError(
            "The .mat file must contain 'data' and 'dataset' variables.")

    return data, dataset


def process_profile(data, dataset, params, model):
    """
    Process the profile data to calculate the dissipation rate.
    """
    profile_num = params['profile_num'] - 1  # Adjust for zero-based indexing
    profile = dataset[profile_num]

    # Extract variables
    P_slow = profile.P_slow
    JAC_T = profile.JAC_T
    JAC_C = profile.JAC_C
    P_fast = profile.P_fast
    W_slow = profile.W_slow
    W_fast = profile.W_fast
    sh1 = profile.sh1
    sh2 = profile.sh2
    Ax = profile.Ax
    Ay = profile.Ay
    T1_fast = profile.T1_fast

    fs_fast = data.fs_fast
    fs_slow = data.fs_slow

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
        sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, P_fast, N2, n, params, fs_fast, model
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

    # Calculate dissipation rate
    diss = get_diss_odas_nagai4gui2024(sh, A, press, info)

    return diss


def save_dissipation_rate(diss, profile_num):
    """
    Save the dissipation rate data to a .mat file.
    """
    filename = f'dissrate_profile_{profile_num}.mat'
    scipy.io.savemat(filename, {'diss': diss})
    print(f"Dissipation rate saved to {filename}")


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
        'profile_num': 1,       # Profile number to process
        'P_start': 0.0,         # Start pressure (dbar)
        'P_end': 1000.0         # End pressure (dbar)
    }
    FILENAME = get_file()
    data, dataset = load_mat_file(FILENAME)

    diss = process_profile(data, dataset, params)

    # Save the results
    save_dissipation_rate(diss, params['profile_num'])

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
