from keras import callbacks, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from predict_dissipation import extract_output_labels, load_output_data, find_K_min
from display_graph import nasmyth, plot_training_history
import numpy as np


import numpy as np

# TODO: IF BAD DATA, handle accordingly for interval pred


def prepare_training_data(data, dataset, params, filename):
    """
    Prepare training data for the CNN model using existing data.
    """
    spectra = []
    scalar_features = []
    integration_ranges = []
    flagoods = []

    num_profiles = len(dataset['P_slow'])

    print(f"Number of profiles available for training: {num_profiles}")

    # Fetch training Data
    for i in range(num_profiles):
        print(f"Processing profile {i+1}/{num_profiles}")
        # DAT_00n_dissrate_0nn.mat
        training_file = f"{filename[:-4]}_dissrate_{i+1:03d}.mat"
        # print(f"Training File: {training_file}")
        diss_data = load_output_data(training_file)
        if (diss_data is None):
            continue
        training_labels = extract_output_labels(diss_data)
        num_windows = training_labels['e'].shape[0]
        num_probes = training_labels['e'].shape[1]
        K_min_values = np.full((num_windows, num_probes), np.nan)

        # Compute all K_min values
        for window_index in range(num_windows):
            for probe_index in range(num_probes):
                epsilon = training_labels['e'][window_index, probe_index]
                nu = training_labels['nu'][window_index]
                P = training_labels['P'][window_index]
                T = training_labels['T'][window_index]
                K = training_labels['K'][window_index, :]
                P_sh = training_labels['sh'][window_index,
                                             probe_index, probe_index, :]
                P_nasmyth = training_labels['Nasmyth_spec'][window_index,
                                                            probe_index, :]
                K_max = training_labels['K_max'][window_index, probe_index]
                flagood = training_labels['flagood'][window_index, probe_index]

                # Validate data
                if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(K_max) or K_max <= K[0]:
                    print(
                        f"Skipping window {window_index}, probe {probe_index} due to invalid data.")
                    continue

                # Compute K_min
                K_min = find_K_min(epsilon, nu, K, P_sh, K_max)
                K_min_values[window_index, probe_index] = K_min

                # Create integration range target
                integration_range = [K_min, K_max]

                # Normalize values
                K_normalized = (K - K.min()) / (K.max() - K.min())
                # Add a small value to avoid log(0)
                P_sh = np.real(P_sh)
                P_sh_log = np.log10(P_sh + 1e-10)
                P_nasmyth_log = np.log10(P_nasmyth + 1e-10)

                spectrum_input = np.stack(
                    (P_sh_log, P_nasmyth_log, K_normalized), axis=-1)

                # Collect scalar features (mean values over the window)
                scalar_feature = np.array([
                    nu,
                    np.mean(P),
                    np.mean(T),
                ])

                # Store data
                spectra.append(spectrum_input)
                scalar_features.append(scalar_feature)
                integration_ranges.append(integration_range)
                flagoods.append(flagood)

    # Convert lists to numpy arrays
    # Shape: (num_samples, spectrum_length, num_channels)
    spectra = np.array(spectra)
    # Shape: (num_samples, num_scalar_features)
    scalar_features = np.array(scalar_features)
    integration_ranges = np.array(
        integration_ranges)  # Shape: (num_samples, 2)
    flagoods = np.array(flagoods)

    return spectra, scalar_features, integration_ranges, flagoods


def create_cnn_model(spectrum_input_shape, scalar_input_shape):
    # Spectral Input
    spectrum_input = layers.Input(
        shape=spectrum_input_shape, name='spectrum_input')
    x = layers.Conv1D(64, kernel_size=5, activation='relu',
                      padding='same')(spectrum_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Scalar Input
    scalar_input = layers.Input(shape=scalar_input_shape, name='scalar_input')

    # Combine Features
    combined = layers.concatenate([x, scalar_input])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(64, activation='relu')(x)

    # Output Layers
    output_integration = layers.Dense(
        2, activation='linear', name='integration_output')(x)
    output_flagood = layers.Dense(
        1, activation='sigmoid', name='flagood_output')(x)

    # Define Model
    model = models.Model(inputs=[spectrum_input, scalar_input], outputs={
        'integration_output': output_integration,
        'flagood_output': output_flagood
    })
    return model


def train_cnn_model(data, dataset, params, filename, existing_model=None):
    # Prepare training data
    spectra, scalar_features, integration_ranges, flagoods = prepare_training_data(
        data, dataset, params, filename)

    # Lower weight of 'bad' data
    sample_weights = np.where(flagoods == 1, 1.0, 0.1)

    if len(spectra) == 0:
        print("No training data was collected, there was an error in preparing the data")
        return

    # Split data into training and validation sets
    X_spectrum_train, X_spectrum_val, X_scalar_train, X_scalar_val, y_integration_train, y_integration_val, y_flagood_train, y_flagood_val, sample_weights_train, sample_weight_val = train_test_split(
        spectra, scalar_features, integration_ranges, flagoods, sample_weights, test_size=0.2, random_state=42
    )

    # Reshape y_flagood to have shape (batch_size, 1)
    y_flagood_train = y_flagood_train.reshape(-1, 1)
    y_flagood_val = y_flagood_val.reshape(-1, 1)

    # Print shapes for verification
    print(f"X_spectrum_train shape: {X_spectrum_train.shape}")
    print(f"X_scalar_train shape: {X_scalar_train.shape}")
    print(f"y_integration_train shape: {y_integration_train.shape}")
    print(f"y_flagood_train shape: {y_flagood_train.shape}")

    if existing_model is not None:
        model = existing_model
        print("Continuing training of existing model...")
    else:
        # Create CNN model
        spectrum_input_shape = (
            X_spectrum_train.shape[1], X_spectrum_train.shape[2])
        scalar_input_shape = (X_scalar_train.shape[1],)
        model = create_cnn_model(spectrum_input_shape, scalar_input_shape)
        print("New model created.")
        model.summary()

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            'integration_output': 'mean_squared_error',
            'flagood_output': 'binary_crossentropy'
        },
        loss_weights={
            'integration_output': 1.0,
            'flagood_output': 1.0
        },
        metrics={
            'integration_output': ['mae'],
            'flagood_output': ['accuracy']
        }
    )

    # Train the model
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        [X_spectrum_train, X_scalar_train],
        {
            'integration_output': y_integration_train,
            'flagood_output': y_flagood_train
        },
        sample_weight=sample_weights_train,
        validation_data=(
            [X_spectrum_val, X_scalar_val],
            {
                'integration_output': y_integration_val,
                'flagood_output': y_flagood_val
            }
        ),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )

    plot_training_history(history)

    return model
