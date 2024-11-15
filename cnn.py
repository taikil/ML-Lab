from keras import callbacks, layers, models
from sklearn.model_selection import train_test_split
from predict_dissipation import extract_output_labels, load_output_data
from display_graph import nasmyth, plot_training_history
from diss_rate_odas_nagai import get_diss_odas_nagai4gui2024, nasmyth
from helper import compute_buoyancy_frequency, compute_density
import numpy as np


import numpy as np


def prepare_training_data(data, dataset, params, filename):
    """
    Prepare training data for the CNN model using existing data.
    """
    spectra = []
    scalar_features = []
    integration_ranges = []

    fs_fast = float(np.squeeze(data['fs_fast']))
    fs_slow = float(np.squeeze(data['fs_slow']))

    num_profiles = dataset.size

    print(f"Number of profiles available for training: {num_profiles}")

    for i in range(num_profiles):
        print(f"Processing profile {i+1}/{num_profiles}")
        profile = dataset[0, i]  # Access the i-th profile
        # DAT_00n_dissrate_0nn.mat
        training_file = f"{filename[:-4]}_dissrate_{i+1:03d}.mat"
        print(f"Training File: {training_file}")
        diss_data = load_output_data(training_file)
        training_labels = extract_output_labels(diss_data)
        print(training_labels['K_max'].shape)
        pred_K_max = training_labels['K_max']

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

        # Prepare data
        SH = np.column_stack((sh1, sh2))
        A = np.column_stack((Ax, Ay))
        speed = W_fast
        T = T1_fast
        P = P_fast

        sigma_theta_fast = compute_density(
            JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast)
        N2 = compute_buoyancy_frequency(sigma_theta_fast, P_fast)

        # Set parameters for dissipation calculation
        fft_length = int(params['fft_length'] * fs_fast)
        diss_length = int(params['diss_length'] * fs_fast)
        overlap = int(params['overlap'] * fs_fast)
        fit_order = params.get('fit_order', 3)
        f_AA = params.get('f_AA', 98)

        try:
            # Calculate dissipation rates
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
                f_AA=f_AA,
                K_max_pred=pred_K_max
            )
        except Exception as e:
            print(f"Error processing profile {i+1}: {e}")
            continue  # Skip this profile if there's an error

        # Extract spectra and integration ranges
        num_estimates = diss['e'].shape[0]
        print(f"Num estimates training: {num_estimates}")
        num_probes = SH.shape[1]
        for index in range(num_estimates):
            for probe_index in range(num_probes):
                P_sh_clean = diss['sh_clean'][index,
                                              probe_index, probe_index, :]
                K = diss['K'][index, :]  # Wavenumber array
                epsilon = diss['e'][index, probe_index]
                K_max = diss['K_max'][index, probe_index]
                K_min = diss['K_min'][index, probe_index]

                # Generate Nasmyth spectrum with epsilon
                nu = diss['nu'][index, 0]
                P_nasmyth, _ = nasmyth(epsilon, nu, K)

                # Create integration range target
                integration_range = [K_min, K_max]

                # Stack P_sh_clean and P_nasmyth to create multi-channel input
                # Shape: (spectrum_length, 2)
                print(f"P_sh_clean: {P_sh_clean.shape}")
                print(f"P_nasmyth: {P_nasmyth.shape}")
                print(index)
                # TODO, FIX NUMBER OF ROWS, P_NASMYTH IS 0 AFTER 101 ITERATIONS
                spectrum_input = np.stack((P_sh_clean, P_nasmyth), axis=-1)

                # Collect scalar features (mean values over the window)
                scalar_feature = np.array([
                    nu,
                    np.mean(P),
                    np.mean(T),
                    np.mean(diss['N2'][index, probe_index])
                ])

                # Store data
                spectra.append(spectrum_input)
                scalar_features.append(scalar_feature)
                integration_ranges.append(integration_range)

    # Convert lists to numpy arrays
    # Shape: (num_samples, spectrum_length, num_channels)
    spectra = np.array(spectra)
    # Shape: (num_samples, num_scalar_features)
    scalar_features = np.array(scalar_features)
    integration_ranges = np.array(
        integration_ranges)  # Shape: (num_samples, 2)

    return spectra, scalar_features, integration_ranges


def create_cnn_model(spectrum_input_shape, scalar_input_shape):
    # Spectral Input
    spectrum_input = layers.Input(
        shape=spectrum_input_shape, name='spectrum_input')
    x = layers.Conv1D(64, kernel_size=5, activation='relu')(spectrum_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Scalar Input
    scalar_input = layers.Input(shape=scalar_input_shape, name='scalar_input')

    # Combine Features
    combined = layers.concatenate([x, scalar_input])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(64, activation='relu')(x)

    # Output Layer
    output = layers.Dense(2, activation='linear')(x)

    # Define Model
    model = models.Model(inputs=[spectrum_input, scalar_input], outputs=output)
    return model


# TRANSFER LEARNING?
def train_cnn_model(data, dataset, params, filename, existing_model=None):
    """
    Train the CNN model to predict the integration range.
    """
    # Prepare training data
    spectra, scalar_features, integration_ranges = prepare_training_data(
        data, dataset, params, filename)

    if len(spectra) == 0:
        print("No training data was collected, there was an error in preparing the data")
        return

    print(f"Spectra shape: {spectra.shape}")
    print(f"Scalar features shape: {scalar_features.shape}")
    print(f"Integration ranges shape: {integration_ranges.shape}")

    # Split data into training and validation sets
    X_spectrum_train, X_spectrum_val, X_scalar_train, X_scalar_val, y_train, y_val = train_test_split(
        spectra, scalar_features, integration_ranges, test_size=0.2, random_state=42
    )

    if existing_model is not None:
        model = existing_model
        print("Continuing training of existing model...")
    else:
        # Create CNN model
        spectrum_input_shape = (spectra.shape[1], spectra.shape[2])
        scalar_input_shape = (scalar_features.shape[1],)
        model = create_cnn_model(spectrum_input_shape, scalar_input_shape)
        print("New model created.")

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        [X_spectrum_train, X_scalar_train], y_train,
        validation_data=([X_spectrum_val, X_scalar_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Optionally, plot training history
    plot_training_history(history)

    return model
