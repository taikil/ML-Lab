import sys
import numpy as np
import scipy.io
from diss_rate_odas_nagai import *
from get_diss_odas import *
from helper import *
from keras import models, layers, callbacks
from scipy.signal import welch
from scipy.signal.windows import hann
from sklearn.model_selection import train_test_split
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
    n, m = get_profile_indices(
        P_fast, P_slow, params, fs_slow, fs_fast)

    nn = np.where((P_fast[n] > params['P_start']) &
                  (P_fast[n] <= params['P_end']))[0]
    range1 = n[nn]  # Subset of fast data based on pressure

    # Despike and filter shear data
    sh1_HP, sh2_HP = despike_and_filter_sh(
        sh1, sh2, Ax, Ay, range1, fs_fast, params)

    diss = calculate_dissipation_rate(
        sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, P_fast, N2, params, fs_fast, model)

    return diss


def get_profile_indices(P_fast, P_slow, params, fs_slow, fs_fast):
    start_index_slow = 0
    end_index_slow = len(P_slow) - 1
    start_index_fast = 0
    end_index_fast = len(P_fast) - 1

    n = np.arange(start_index_fast, end_index_fast + 1)
    m = np.arange(start_index_slow, int(
        end_index_fast / (fs_fast / fs_slow)) + 1)
    return n, m


def compute_shear_spectrum(shear_signal, fs):
    # Remove mean and apply window function
    shear_signal_detrended = shear_signal - np.mean(shear_signal)
    window = hann(len(shear_signal_detrended))
    shear_windowed = shear_signal_detrended * window

    # Compute FFT
    nfft = len(shear_windowed)
    freq, Pxx = welch(shear_windowed, fs=fs, window='hanning',
                      nperseg=nfft, noverlap=0, scaling='density')

    # Convert frequency to wavenumber (k = f / W)
    # W is the fall rate (speed)
    return freq, Pxx


def calculate_dissipation_rate(sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, P_fast, N2, params, fs_fast, model):
    """
    Calculate the dissipation rate over smaller windows using the CNN-predicted integration range.
    """
    # Prepare variables
    SH = np.column_stack((sh1_HP, sh2_HP))
    A = np.column_stack((Ax, Ay))
    speed = W_fast
    T = T1_fast
    P = P_fast

    # Set parameters for dissipation calculation
    fft_length = int(params['fft_length'] * fs_fast)
    diss_length = int(params['diss_length'] * fs_fast)
    overlap = int(params['overlap'] * fs_fast)
    fit_order = params.get('fit_order', 3)
    f_AA = params.get('f_AA', 98)

    # Estimate epsilon using get_diss_odas_nagai4gui2024
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

    # Initialize lists to store results per window
    e_final_list = []
    best_k_range_list = []
    k_common_list = []
    P_shear_interp_list = []
    P_nasmyth_list = []

    print(f"diss['e'] shape: {diss['e'].shape}")
    print(f"diss['K_max'] shape: {diss['K_max'].shape}")
    print(f"diss['K'] shape: {diss['K'].shape}")
    print(f"diss['K']: {diss['K'][0]}")
    print(f"diss['sh_clean'] shape: {diss['sh_clean'].shape}")

    # Loop over each window
    for index in range(num_estimates):
        # For each window, process each probe
        for probe_index in range(num_probes):
            # Extract data for this window and probe
            epsilon = diss['e'][index, probe_index]
            K_max = diss['K_max'][index, probe_index]
            K = diss['K'][index, :]   # Shape: (1025,)
            P_sh_clean = diss['sh_clean'][index, probe_index, probe_index, :]

            # Interpolate onto common wavenumber grid
            k_common = np.linspace(K.min(), K.max(), 512)
            P_shear_interp = np.interp(k_common, K, P_sh_clean)
            P_shear_interp = np.where(
                P_shear_interp <= 0, 1e-10, P_shear_interp)

            # Prepare input for the CNN
            spectrum_input = P_shear_interp.reshape(
                1, -1, 1)  # Reshape for CNN input

            # Predict integration range using the CNN
            predicted_range = model.predict(spectrum_input)
            K_min_pred, K_max_pred = predicted_range[0]

            # Validate predicted range
            K_min_pred = max(K[0], K_min_pred)
            K_max_pred = min(K[-1], K_max_pred)
            if K_min_pred >= K_max_pred:
                print(
                    f"Invalid predicted integration range at window {index}, probe {probe_index}. Using default range.")
                K_min_pred, K_max_pred = K[0], K_max

            # kinematic viscosity
            nu = visc35(np.mean(T))

            # Generate the Nasmyth spectrum with epsilon
            P_nasmyth = nasmyth_spectrum(k_common, epsilon, nu)
            P_nasmyth = np.where(P_nasmyth <= 0, 1e-10, P_nasmyth)

            # Calculate the final dissipation rate using the CNN-predicted integration range
            idx_integration = np.where(
                (k_common >= K_min_pred) & (k_common <= K_max_pred))[0]
            e_final = 7.5 * nu * \
                np.trapz(P_shear_interp[idx_integration],
                         k_common[idx_integration])

            print(
                f"Window {index}, Probe {probe_index}, Final dissipation rate after CNN integration range: {e_final:.2e} W/kg")

            e_final_list.append(e_final)
            best_k_range_list.append([K_min_pred, K_max_pred])
            k_common_list.append(k_common)
            P_shear_interp_list.append(P_shear_interp)
            P_nasmyth_list.append(P_nasmyth)

            plot_spectra(k_common, P_shear_interp, P_nasmyth,
                         [K_min_pred, K_max_pred], e_final, window_index=index, probe_index=probe_index)

    diss_results = {
        'e_final': e_final_list,
        'best_k_range': best_k_range_list,
        'k_common': k_common_list,
        'P_shear_interp': P_shear_interp_list,
        'P_nasmyth': P_nasmyth_list,
        'num_estimates': num_estimates,
        'num_probes': num_probes,
    }

    return diss_results


def train_cnn_model(data, dataset, params):
    """
    Train the CNN model to predict the integration range.
    """
    # Prepare training data
    spectra, integration_ranges = prepare_training_data(data, dataset, params)

    print(f"Spectra len: {len(spectra)}")
    print(f"IR len: {len(integration_ranges)}")

    # Convert lists to numpy arrays
    X = np.array(spectra)  # Shape: (samples, spectrum_length)
    y = np.array(integration_ranges)  # Shape: (samples, 2)

    # Reshape X for CNN input
    X = X.reshape(-1, X.shape[1], 1)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create CNN model
    input_shape = (X.shape[1], 1)
    model = create_cnn_model(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Optionally, plot training history
    plot_training_history(history)

    return model


def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=5,
              activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, kernel_size=5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(256, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(128, activation='relu'))
    # Output layer for predicting integration range (two values)
    model.add(layers.Dense(2, activation='linear'))
    return model


def prepare_training_data(data, dataset, params):
    """
    Prepare training data for the CNN model using existing data.
    """
    spectra = []
    integration_ranges = []

    fs_fast = np.squeeze(data['fs_fast'])
    fs_slow = np.squeeze(data['fs_slow'])

    fs_fast = float(fs_fast)
    fs_slow = float(fs_slow)

    num_profiles = dataset.size

    print(f"Number of profiles available for training: {num_profiles}")

    for i in range(num_profiles):
        print(f"Processing profile {i+1}/{num_profiles}")
        profile = dataset[0, i]  # Access the i-th profile

        # Extract variables for the profile
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
            print(f"Missing expected field in profile {i+1}: {e}")
            continue

        # Prepare data for dissipation calculation
        SH = np.column_stack((sh1, sh2))
        A = np.column_stack((Ax, Ay))
        speed = W_fast
        T = T1_fast
        P = P_fast

        # Set parameters for dissipation calculation
        fft_length = int(params['fft_length'] * fs_fast)
        diss_length = int(params['diss_length'] * fs_fast)
        overlap = int(params['overlap'] * fs_fast)
        fit_order = params.get('fit_order', 3)
        f_AA = params.get('f_AA', 98)
        # fit_2_isr = params.get('fit_2_isr', 1.5e-5)
        # f_limit = params.get('f_limit', np.inf)

        # Call the dissipation calculation function
        try:
            diss = get_diss_odas_nagai4gui2024(
                SH=SH,
                A=A,
                fft_length=fft_length,
                diss_length=diss_length,
                overlap=overlap,
                fs=fs_fast,
                speed=speed,
                T=T,
                N2=P,
                P=P,
                fit_order=fit_order,
                f_AA=f_AA,
            )
        except Exception as e:
            print(f"Error processing profile {i+1}: {e}")
            continue  # Skip this profile if there's an error

        # Extract spectra and integration ranges
        num_estimates = diss['e'].shape[1]
        num_probes = SH.shape[1]
        for index in range(num_estimates):
            for probe_index in range(num_probes):
                P_sh_clean = diss['sh_clean'][:,
                                              probe_index, probe_index, index]
                K = diss['K'][:, index]
                epsilon = diss['e'][probe_index, index]
                K_max = diss['K_max'][probe_index, index]

                # Use K_max as the upper limit of the integration range
                K_min = K[0]
                integration_range = [K_min, K_max]

                # Interpolate the spectrum onto a common wavenumber grid
                k_common = np.linspace(K[0], K[-1], 512)
                P_shear_interp = np.interp(k_common, K, P_sh_clean)

                # Store the interpolated spectrum and integration range
                spectra.append(P_shear_interp)
                integration_ranges.append(integration_range)

    # Convert lists to numpy arrays
    spectra = np.array(spectra)
    integration_ranges = np.array(integration_ranges)

    return spectra, integration_ranges


def plot_training_history(history):
    plt.figure(figsize=(10, 4))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_dissipation_rate(diss_results, profile_num):
    """
    Save the dissipation rate data to a .mat file.
    """
    filename = f'dissrate_profile_{profile_num}.mat'
    scipy.io.savemat(filename, diss_results)
    print(f"Dissipation rate saved to {filename}")


def plot_spectra(k_obs, P_shear_obs, P_nasmyth, best_k_range, best_epsilon, window_index=None, probe_index=None):
    plt.figure(figsize=(10, 6))
    plt.loglog(k_obs, P_shear_obs, label='Observed Shear Spectrum')
    plt.loglog(k_obs, P_nasmyth,
               label=f'Nasmyth Spectrum (ε={best_epsilon:.2e} W/kg)')
    plt.axvline(best_k_range[0], color='r', linestyle='--',
                label='Integration Range Start')
    plt.axvline(best_k_range[1], color='g',
                linestyle='--', label='Integration Range End')
    plt.xlabel('Wavenumber (cpm)')
    plt.ylabel('Shear Spectrum [(s$^{-1}$)$^2$/cpm]')
    if window_index is not None and probe_index is not None:
        plt.title(
            f'Shear Spectrum Fit (Window {window_index}, Probe {probe_index})')
    else:
        plt.title('Shear Spectrum Fit')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'spectrum_window_{window_index}_probe_{probe_index}.png')
    plt.close()


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
        'fit_order': 3,         # Order of polynomial fit
        'f_AA': 98,             # Anti-aliasing filter frequency (Hz)
        # Threshold for inertial subrange fitting (W/kg)
        'fit_2_isr': 1.5e-5,
        # Maximum frequency to use when estimating dissipation (Hz)
        'f_limit': np.inf,
        'min_duration': 60.0,   # Minimum profile duration (s)
        # Profile number to process (fixed to 1 for naming)
        'profile_num': 1,
        'P_start': 0.0,         # Start pressure (dbar)
        'P_end': 1000.0         # End pressure (dbar)
    }
    FILENAME = get_file()
    data, dataset = load_mat_file(FILENAME)

    # Load or train the CNN model for integration range prediction
    model_filename = input(
        f"Enter desired model filename: ")
    model_filename = f'{model_filename}.keras'
    try:
        # Try to load the pre-trained model
        model = models.load_model(model_filename)
        print(f"Loaded pre-trained CNN model from {model_filename}")
    except (IOError, OSError, ValueError):
        # If model file does not exist, train the model
        print(f"No pre-trained model found. Training a new CNN model.")
        model = train_cnn_model(data, dataset, params)
        # Save the trained model
        model.save(model_filename)
        print(f"Saved trained CNN model to {model_filename}")

    num_profiles = dataset['P_slow'].shape[1]
    print(f"Number of profiles in dataset: {num_profiles}")
    # Loop over all profiles or prompt for specific profile
    profile_num = int(
        input(f"Enter profile number to process (0 to {num_profiles - 1}): "))

    diss = process_profile(data, dataset, params, profile_num, model)

    save_dissipation_rate(diss, profile_num)

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
