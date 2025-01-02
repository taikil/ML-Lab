from keras import models
import sys
import numpy as np
import scipy.io
from scipy.integrate import cumtrapz
from scipy.optimize import brentq
from diss_rate_odas_nagai import *
from helper import *
from keras import models
from display_graph import *
from cnn import *
import mat73


def get_file():
    if len(sys.argv) == 2:
        filename = str(sys.argv[1])
        return filename
    else:
        print("Invalid Input; Provide the .mat filename as a command-line argument.")
        print("Exiting Program...")
        sys.exit()


def load_mat_file(filename):
    print("Loading .mat file using mat73.")
    mat_contents = mat73.loadmat(filename)

    print("Keys in mat_contents:", mat_contents.keys())

    data = mat_contents.get('data', None)
    dataset = mat_contents.get('dataset', None)

    if data is not None and dataset is not None:
        return data, dataset
    else:
        raise ValueError(
            "The .mat file does not contain 'data' and 'dataset' variables.")


def process_profile(data, dataset, params, profile_num=0, model=None):
    """
    Process the profile data to calculate the dissipation rate.
    """
    # Extract sampling frequencies from 'data'
    fs_fast = np.squeeze(data['fs_fast'])
    fs_slow = np.squeeze(data['fs_slow'])

    fs_fast = float(fs_fast)
    fs_slow = float(fs_slow)

    # Get the number of profiles
    num_profiles = len(dataset['P_slow'])
    if profile_num >= num_profiles:
        raise ValueError(
            f"Profile number {profile_num} exceeds available profiles ({num_profiles}).")

    # Extract variables for the selected profile
    try:
        P_slow = np.squeeze(dataset['P_slow'][profile_num])
        JAC_T = np.squeeze(dataset['JAC_T'][profile_num])
        JAC_C = np.squeeze(dataset['JAC_C'][profile_num])
        P_fast = np.squeeze(dataset['P_fast'][profile_num])
        W_slow = np.squeeze(dataset['W_slow'][profile_num])
        W_fast = np.squeeze(dataset['W_fast'][profile_num])
        sh1 = np.squeeze(dataset['sh1'][profile_num])
        sh2 = np.squeeze(dataset['sh2'][profile_num])
        Ax = np.squeeze(dataset['Ax'][profile_num])
        Ay = np.squeeze(dataset['Ay'][profile_num])
        T1_fast = np.squeeze(dataset['T1_fast'][profile_num])
    except KeyError as e:
        raise KeyError(f"Missing expected field in dataset: {e}")

    # Compute derived quantities
    sigma_theta_fast = compute_density(
        JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast)
    N2 = compute_buoyancy_frequency(sigma_theta_fast, P_fast)

    # Prepare profile indices
    n, m = get_profile_indices(P_slow, W_slow, params, fs_slow, fs_fast)

    nn = np.where((P_fast[n] > params['P_start']) &
                  (P_fast[n] <= params['P_end']))[0]
    range1 = n[nn]  # Subset of fast data based on pressure

    sh1_HP, sh2_HP = despike_and_filter_sh(
        sh1, sh2, Ax, Ay, range1, fs_fast, params)

    diss = calculate_dissipation_rate(
        sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, P_fast, N2, params, fs_fast, range1, model)

    return diss


def load_output_data(output_filename):
    """
    Load output labels from the .mat file.
    """
    try:
        mat_contents = scipy.io.loadmat(
            output_filename, struct_as_record=False, squeeze_me=True)
        diss = mat_contents.get('diss', None)
        if diss is None:
            raise ValueError(
                "The output .mat file does not contain 'diss' variable.")
        print("Output labels loaded successfully.")
        return diss
    except NotImplementedError:
        # Fallback to hdf5storage for MATLAB v7.3 files
        print("Loading .mat file using hdf5storage.")
        mat_contents = mat73.loadmat(output_filename)
        diss = mat_contents.get('diss', None)
        if diss is None:
            raise ValueError(
                "The .mat file does not contain both 'data' and 'dataset' variables.")
        return diss

    except FileNotFoundError:
        print(f"File: {output_filename} does not exist, skipping profile.")
        return None


def extract_output_labels(diss):
    """
    Extract output labels from the dissipation data for a given profile number.
    """
    # List of variables to extract from 'diss'
    output_variables = ['e', 'nu', 'sh',
                        'K_max', 'K', 'Nasmyth_spec', 'flagood', 'P', 'T']

    # Dictionary to store the extracted variables
    output_labels = {}

    for var_name in output_variables:
        try:
            var_data = getattr(diss, var_name)
            if isinstance(var_data, np.ndarray) and var_data.ndim > 1:
                print(f"Shape: {var_name} : {var_data.shape}")
                # var_data = var_data.flatten()
            output_labels[var_name] = var_data
        except AttributeError:
            raise AttributeError(
                f"Variable '{var_name}' not found in 'diss' structure.")
        except Exception as e:
            raise Exception(f"Error extracting variable '{var_name}': {e}")

    return output_labels


def find_K_min(epsilon, nu, K, P_shear, K_max):
    # Ensure K and P_shear are numpy arrays
    K = np.array(K)
    P_shear = np.array(P_shear)

    # Sort K and P_shear in ascending order of K
    sorted_indices = np.argsort(K)
    K = K[sorted_indices]
    P_shear = P_shear[sorted_indices]

    # Compute cumulative integral from smallest K up to K_max
    cumulative_integral = cumtrapz(P_shear, K, initial=0)

    # Interpolate cumulative integral for precise calculations
    from scipy.interpolate import interp1d
    integral_interp = interp1d(
        K, cumulative_integral, kind='linear', fill_value="extrapolate")

    # Define the function to find the root
    def func(K_min):
        # Compute integral from K_min to K_max
        integral = integral_interp(K_max) - integral_interp(K_min)
        # Compute the difference between calculated epsilon and known epsilon
        return 7.5 * nu * integral - epsilon

    # Initial bounds for K_min
    K_min_lower = K[0]
    K_min_upper = K_max

    # Check if the function changes sign over the interval, if not K_min will be 0
    if func(K_min_lower) * func(K_min_upper) > 0:
        return 0

    # Solve for K_min
    K_min_solution = brentq(func, K_min_lower, K_min_upper)

    return K_min_solution


def get_profile_indices(P_slow, W_slow, params, fs_slow, fs_fast):
    profile = get_profile(P_slow, W_slow, params['P_min'], params['W_min'],
                          params['direction'], params['min_duration'], fs_slow)

    start_index_slow = profile[0, 0] - 1
    end_index_slow = profile[1, 0] - 1
    start_index_fast = int(1 + round((fs_fast/fs_slow)*(start_index_slow)))
    end_index_fast = int(round((fs_fast/fs_slow)*(end_index_slow)))
    n = np.arange(start_index_fast, end_index_fast + 1)
    m = np.arange(start_index_slow, end_index_slow + 1)
    return n, m


def calculate_dissipation_rate(sh1, sh2, Ax, Ay, T1_fast, W_fast, P_fast, N2, params, fs_fast, range1, model):
    """
    Calculate the dissipation rate over smaller windows using the CNN-predicted integration range.
    """
    # Prepare variables
    SH = np.column_stack((sh1, sh2))
    A = np.column_stack((Ax[range1], Ay[range1]))
    speed = W_fast[range1]
    T = T1_fast[range1]
    P = P_fast[range1]

    # np.savetxt("sh_diss.txt", SH, fmt='%.6e')

    # Set parameters for dissipation calculation
    fft_length = int(params['fft_length'] * fs_fast)
    diss_length = int(params['diss_length'] * fs_fast)
    overlap = int(params['overlap'] * fs_fast)
    fit_order = params.get('fit_order', 3)
    f_AA = params.get('f_AA', 98)

    # Initial estimate epsilon
    diss = get_diss_odas_nagai4gui2024(
        SH=SH,
        A=A,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        fs=fs_fast,
        speed=speed,
        T=T,
        N2=N2,
        P=P,
        fit_order=fit_order,
        f_AA=f_AA
    )

    # Number of estimates (windows)
    num_estimates = diss['e'].shape[0]  # !!!!!!
    num_probes = SH.shape[1]

    print(f"Diss size: {num_estimates}")
    spectra_data = []

    # Loop over each window
    for index in range(num_estimates):
        for probe_index in range(num_probes):
            # Extract data for this window and probe
            epsilon = diss['e'][index, probe_index]
            K = diss['K'][index, :]   # Shape: (1025,)
            P_sh = diss['sh'][index, probe_index, probe_index, :]

            # Prepare input for the CNN
            spectrum_input = P_sh.reshape(
                1, -1, 1)  # Reshape for CNN input

            # Generate Nasmyth spectrum with epsilon
            nu = diss['nu'][index, 0]
            P_nasmyth, _ = nasmyth(epsilon, nu, K)

            # Normalize
            K_normalized = (K - K.min()) / (K.max() - K.min())
            P_sh = np.real(P_sh)
            P_sh_log = np.log10(P_sh + 1e-10)
            P_nasmyth_log = np.log10(P_nasmyth + 1e-10)

            # Prepare spectral input
            # Shape: (spectrum_length, 2)
            print(f"P_sh_log shape: {P_sh_log.shape}")
            print(f"nasmyth log shape: {P_nasmyth_log.shape}")
            print(f"K_normalized shape: {K_normalized.shape}")
            spectrum_input = np.stack(
                (P_sh_log, P_nasmyth_log, K_normalized), axis=-1)
            # Add batch dimension
            spectrum_input = spectrum_input[np.newaxis, ...]

            # Prepare scalar features
            scalar_feature = np.array([
                nu,
                np.mean(P),
                np.mean(T),
            ])
            # Add batch dimension
            scalar_feature = scalar_feature[np.newaxis, :]

            # Predict integration range using the CNN
            outputs = model.predict(
                [spectrum_input, scalar_feature])
            predicted_range = outputs['integration_output']
            flagood_pred = outputs['flagood_output']
            K_min_pred, K_max_pred = predicted_range[0]
            flagood_pred = flagood_pred[0][0]

            diss['K_min'][index, probe_index] = K_min_pred
            diss['K_max'][index, probe_index] = K_max_pred
            # if flagood_pred <= 0.1:
            #     diss['flagood'][index, probe_index] = 0
            # else:
            #     diss['flagood'][index, probe_index] = 1

            diss['flagood'][index, probe_index] = flagood_pred

            # kinematic viscosity
            nu = diss['nu'][index, 0]

            # Calculate the final dissipation rate using the CNN-predicted integration range
            print(f"K_MAX PRED FOR INTEGRATION: {K_max_pred}")
            print(f"K_MIN PRED FOR INTEGRATION: {K_min_pred}")
            idx_integration = np.where(
                (K >= K_min_pred) & (K <= K_max_pred))[0]
            epsilon_cnn = 7.5 * nu * \
                np.trapz(P_sh[idx_integration],
                         K[idx_integration])

            # Update epsilon in the diss dictionary
            diss['e'][index, probe_index] = epsilon_cnn

            # Update Nasmyth spectrum in the diss dictionary
            P_nasmyth, _ = nasmyth(epsilon_cnn, nu, K)
            diss['Nasmyth_spec'][index, probe_index, :] = P_nasmyth

            # Update Krho and other derived quantities
            N2_mean = diss['N2'][index, probe_index]
            diss['Krho'][index, probe_index] = 0.2 * epsilon_cnn / N2_mean

            print(
                f"Window {index}, Probe {probe_index}, Final dissipation rate after CNN integration range: {epsilon_cnn:.2e} W/kg")

            K = diss['K'][index, :]
            P_nasmyth = diss['Nasmyth_spec'][index, probe_index, :]
            epsilon_cnn = diss['e'][index, probe_index]
            K_min_pred = diss['K_min'][index, probe_index]
            K_max_pred = diss['K_max'][index, probe_index]

            # Prepare data for plotting
            plot_data = {
                'k_obs': K,
                'P_shear_obs': P_sh,
                'P_nasmyth': P_nasmyth,
                'k_min': K_min_pred,
                'k_max': K_max_pred,
                'best_epsilon': epsilon_cnn,
                'window_index': index,
                'probe_index': probe_index,
                'nu': nu
            }
            spectra_data.append(plot_data)

    # Plot the spectra interactively
    # TODO add return of updated data
    plot_spectra_interactive(spectra_data)
    final_data = plot_spectra_interactive.saved_data

    for item in final_data:
        i = item['window_index']
        p = item['probe_index']
        diss['K_min'][i, p] = item['k_min']
        diss['K_max'][i, p] = item['k_max']
        diss['e'][i, p] = item['best_epsilon']
        diss['Nasmyth_spec'][i, p, :] = item['P_nasmyth']

    return diss


def save_dissipation_rate(filename, diss_results, profile_num):
    """
    Save the dissipation rate data to a .mat file.
    """
    filename = f'{filename[:-4]}_dissrate_profile_{profile_num}.mat'
    data = {'diss': diss_results}
    scipy.io.savemat(filename, data)
    print(f"Dissipation rate saved to {filename}")


def main():
    params = {
        'HP_cut': 0.2,          # High-pass filter cutoff frequency (Hz)
        'LP_cut': 6.0,         # Low-pass filter cutoff frequency (Hz)
        'P_min': 15.0,           # Minimum pressure (dbar)
        'W_min': 0.1,           # Minimum profiling speed (m/s)
        'direction': 'down',    # Profiling direction
        'fft_length': 4.0,      # FFT length (s)
        'diss_length': 8.0,     # Dissipation length (s)
        'overlap': 4.0,         # Overlap length (s)
        'fit_order': 3,         # Order of polynomial fit
        'f_AA': 98,             # Anti-aliasing filter frequency (Hz)
        # Threshold for inertial subrange fitting (W/kg)
        'fit_2_isr': 1.5e-5,
        # Maximum frequency to use when estimating dissipation (Hz)
        'f_limit': np.inf,
        'min_duration': 20.0,   # Minimum profile duration (s)
        # Profile number to process (fixed to 1 for naming)
        'profile_num': 1,
        'P_start': 10.0,         # Start pressure (dbar)
        'P_end': 1000.0         # End pressure (dbar)
    }
    FILENAME = get_file()
    data, dataset = load_mat_file(FILENAME)

    # Load or train the CNN model for integration range prediction
    model_filename = input(
        f"Enter desired model filename: ")
    model_filename = f'{model_filename}.keras'
    while True:
        choice = input(
            "Do you want to \n(1) continue training the existing model\n(2) train a new model\n(3) use the existing model without retraining?\nEnter 1, 2, or 3: ")
        if choice in ['1', '2', '3']:
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    if choice == '1':
        # Try to load the pre-trained model
        try:
            model = models.load_model(model_filename)
            print(f"Loaded existing CNN model from {model_filename}")
            # Continue training the model
            model = train_cnn_model(
                data, dataset, params, FILENAME, existing_model=model)
            # Save the updated model
            if model is None:
                print("No model was returned by train_cnn_model.")
                sys.exit(1)
            model.save(model_filename)
            print(f"Updated CNN model saved to {model_filename}")
        except (IOError, OSError, ValueError) as e:
            print(f"Failed to load the model from {model_filename}: {e}")
            print("Training a new model instead.")
            model = train_cnn_model(data, dataset, params, FILENAME)
            if model is None:
                print("No model was returned by train_cnn_model.")
                sys.exit(1)
            model.save(model_filename)
    elif choice == '2':
        # Train a new model
        print("Training a new CNN model.")
        model = train_cnn_model(data, dataset, params, FILENAME)
        if model is None:
            print("No model was returned by train_cnn_model.")
            sys.exit(1)
        model.save(model_filename)
        print(f"New CNN model saved to {model_filename}")
    else:
        # Use the existing model without retraining
        try:
            model = models.load_model(model_filename)
            print(f"Loaded existing CNN model from {model_filename}")
            model.summary()
        except (IOError, OSError, ValueError) as e:
            print(f"Failed to load the model from {model_filename}: {e}")
            sys.exit("Cannot proceed without a valid model.")

    num_profiles = len(dataset['P_slow'])
    print(f"Number of profiles in dataset: {num_profiles}")
    # Loop over all profiles or prompt for specific profile
    profile_num = int(
        input(f"Enter profile number to process (1 to {num_profiles}): "))
    profile_num -= 1

    diss = process_profile(data, dataset, params, profile_num, model)

    save_dissipation_rate(FILENAME, diss, profile_num + 1)

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
