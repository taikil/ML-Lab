import sys
import numpy as np
import scipy.io
from keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import hdf5storage
from classifier import *


def get_input_output_files():
    if len(sys.argv) == 3:
        input_filename = str(sys.argv[1])
        output_filename = str(sys.argv[2])
        return input_filename, output_filename
    else:
        print("Invalid Input; Provide the input and output .mat filenames as command-line arguments.")
        print("Usage: python script.py input_file.mat output_file.mat")
        sys.exit()


def load_input_data(input_filename):
    try:
        mat_contents = scipy.io.loadmat(
            input_filename, struct_as_record=False, squeeze_me=True)
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
        mat_contents = hdf5storage.loadmat(input_filename)
        data = mat_contents.get('data', None)
        dataset = mat_contents.get('dataset', None)
        if data is None or dataset is None:
            raise ValueError(
                "The .mat file does not contain both 'data' and 'dataset' variables.")
        return data, dataset


def load_output_labels(output_filename):
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
        mat_contents = hdf5storage.loadmat(output_filename)
        diss = mat_contents.get('diss', None)
        if diss is None:
            raise ValueError(
                "The .mat file does not contain both 'data' and 'dataset' variables.")
        return diss


def extract_input_features(data, dataset, profile_num=0):
    """
    Extract input features from the raw data for a given profile number.
    """
    # Extract sampling frequencies
    fs_fast = float(np.squeeze(data['fs_fast']))
    fs_slow = float(np.squeeze(data['fs_slow']))

    # Extract variables
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

    # Compute derived quantities using existing functions
    sigma_theta_fast = compute_density(
        JAC_T, JAC_C, P_slow, P_fast, fs_slow, fs_fast)
    N2 = compute_buoyancy_frequency(sigma_theta_fast, P_fast)

    # Process shear signals
    n = np.arange(len(P_fast))  # Use full range or adjust as needed
    sh1_HP, sh2_HP = despike_and_filter_sh(
        sh1, sh2, Ax, Ay, n, fs_fast, params={'HP_cut': 1.0})

    # Ensure all variables have the same length
    min_length = min(len(sh1_HP), len(sh2_HP), len(
        Ax), len(Ay), len(T1_fast), len(W_fast), len(N2))
    sh1_HP = sh1_HP[:min_length]
    sh2_HP = sh2_HP[:min_length]
    Ax = Ax[:min_length]
    Ay = Ay[:min_length]
    T1_fast = T1_fast[:min_length]
    W_fast = W_fast[:min_length]
    N2 = N2[:min_length]

    # Stack variables to form the input data
    input_features = np.column_stack(
        (sh1_HP, sh2_HP, Ax, Ay, T1_fast, W_fast, N2))

    return input_features


def extract_output_labels(diss, profile_num=0):
    """
    Extract output labels from the dissipation data for a given profile number.
    """
    # List of variables to extract from 'diss'
    output_variables = [
        'e', 'K_max', 'warning', 'method', 'Nasmyth_spec', 'sh',
        'sh_clean', 'AA', 'UA', 'F', 'K', 'speed', 'nu', 'P', 'T', 'flagood'
    ]

    # Dictionary to store the extracted variables
    output_labels = {}

    for var_name in output_variables:
        try:
            var_data = getattr(diss, var_name)
            if isinstance(var_data, np.ndarray) and var_data.ndim > 1:
                # Assuming the first dimension corresponds to profiles
                var_profile = var_data[profile_num]
            else:
                var_profile = var_data  # Scalar or 1D array

            # Flatten if necessary
            if isinstance(var_profile, np.ndarray):
                var_profile = var_profile.flatten()

            output_labels[var_name] = var_profile
        except AttributeError:
            raise AttributeError(
                f"Variable '{var_name}' not found in 'diss' structure.")
        except Exception as e:
            raise Exception(f"Error extracting variable '{var_name}': {e}")

    return output_labels


def align_input_output(input_features, output_labels):
    """
    Align input features and output labels.
    """
    # Find the minimum length among input features and all output labels
    num_input_samples = input_features.shape[0]
    print(f"input_features len: {len(input_features)}")
    print(f"output features len: {len(output_labels)}")
    min_samples = num_input_samples

    # Determine the minimum length across all output labels
    for var_name, var_data in output_labels.items():
        if isinstance(var_data, np.ndarray):
            min_samples = min(min_samples, len(var_data))

    # Trim input features
    input_features = input_features[:min_samples]

    # Trim each output label
    for var_name in output_labels:
        var_data = output_labels[var_name]
        if isinstance(var_data, np.ndarray):
            output_labels[var_name] = var_data[:min_samples]
        else:
            # If scalar, replicate to match min_samples
            output_labels[var_name] = np.full((min_samples,), var_data)

    return input_features, output_labels


def preprocess_data(input_features, output_labels):
    """
    Preprocess input features and output labels.
    """
    # Handle NaNs in input features
    nan_indices = np.isnan(input_features).any(axis=1)

    # Combine NaN indices from all output labels
    for var_name, var_data in output_labels.items():
        if isinstance(var_data, np.ndarray):
            nan_indices |= np.isnan(var_data)

    # Filter out samples with NaNs
    input_features = input_features[~nan_indices]
    for var_name in output_labels:
        var_data = output_labels[var_name]
        if isinstance(var_data, np.ndarray):
            output_labels[var_name] = var_data[~nan_indices]
        else:
            # Scalars remain unchanged
            pass

    # Scale input features
    scaler = StandardScaler()
    input_features_scaled = scaler.fit_transform(input_features)
    print(
        f"Length of input features after preprocessing: {len(input_features_scaled)}")

    # Preprocess output labels
    for var_name in output_labels:
        var_data = output_labels[var_name]
        if var_name == 'e':
            # Log-transform 'e' to handle large range
            output_labels[var_name] = np.log10(var_data + 1e-10)
        elif isinstance(var_data, np.ndarray) and var_data.dtype.kind in 'fi':
            # Standardize numerical variables
            output_labels[var_name] = (
                var_data - np.nanmean(var_data)) / np.nanstd(var_data)
        else:
            # Handle categorical variables if necessary
            pass

    return input_features_scaled, output_labels, scaler


def create_sequences(features, labels_dict, seq_length):
    X = []
    y_dict = {key: [] for key in labels_dict.keys()}

    if len(features) <= seq_length:
        print("Not enough data to create sequences.")
        return np.array([]), {key: np.array([]) for key in labels_dict.keys()}

    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        for key in labels_dict.keys():
            y_dict[key].append(labels_dict[key][i:i+seq_length])

    # Convert lists to arrays
    X = np.array(X)
    for key in y_dict.keys():
        y_dict[key] = np.array(y_dict[key])

    return X, y_dict


def build_encoder_decoder_model(num_features, seq_length, output_labels):
    """
    Build an encoder-decoder model for sequence-to-sequence prediction with multiple outputs.
    """
    # Encoder
    encoder_inputs = layers.Input(shape=(seq_length, num_features))
    encoder_lstm = layers.LSTM(128, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = layers.Input(shape=(seq_length, num_features))
    decoder_lstm = layers.LSTM(128, return_sequences=True)
    decoder_outputs = decoder_lstm(
        decoder_inputs, initial_state=encoder_states)

    # Output layers for each output label
    output_layers = {}
    for var_name in output_labels.keys():
        if var_name in ['warning', 'method', 'flagood']:
            # Categorical variables - use appropriate activation
            # Assuming binary classification for simplicity
            output_layer = layers.Dense(
                seq_length, activation='sigmoid', name=var_name)(decoder_outputs)
        elif var_name in ['Nasmyth_spec', 'sh', 'sh_clean', 'AA', 'UA', 'F', 'K']:
            # Complex structures - output appropriate shapes
            # For now, we'll output a fixed-size vector per time step
            output_layer = layers.TimeDistributed(
                layers.Dense(10), name=var_name)(decoder_outputs)
        else:
            # Continuous variables - regression output
            output_layer = layers.TimeDistributed(
                layers.Dense(1), name=var_name)(decoder_outputs)
        output_layers[var_name] = output_layer

    # Model
    model = models.Model([encoder_inputs, decoder_inputs],
                         list(output_layers.values()))

    # Compile the model with appropriate loss functions
    losses = {}
    for var_name in output_labels.keys():
        if var_name in ['warning', 'method', 'flagood']:
            # Binary cross-entropy loss for classification
            losses[var_name] = 'binary_crossentropy'
        else:
            # Mean squared error for regression
            losses[var_name] = 'mse'

    model.compile(optimizer='adam', loss=losses)

    print("Encoder-decoder model with multiple outputs built successfully.")
    return model


def train_model(model, X_train_enc, X_train_dec, y_train_dict, X_val_enc, X_val_dec, y_val_dict, epochs=20):
    """
    Train the encoder-decoder model.
    """
    model.fit([X_train_enc, X_train_dec], y_train_dict,
              epochs=epochs,
              batch_size=32,
              validation_data=([X_val_enc, X_val_dec], y_val_dict))
    print("Model training completed.")


def main():

    input_filename, output_filename = get_input_output_files()
    data, dataset = load_input_data(input_filename)
    diss = load_output_labels(output_filename)

    num_profiles = len(dataset['P_slow'])

    all_input_features = []
    all_output_labels_list = []

    for profile_num in range(num_profiles):
        input_features = extract_input_features(
            data, dataset, profile_num)
        output_labels = extract_output_labels(diss, profile_num)

        input_features_aligned, output_labels_aligned = align_input_output(
            input_features, output_labels)

        all_input_features.append(input_features_aligned)
        all_output_labels_list.append(output_labels_aligned)

    # Concatenate all data
    input_features = np.concatenate(all_input_features, axis=0)

    # Combine all output labels
    combined_output_labels = {}
    for var_name in all_output_labels_list[0].keys():
        combined_output_labels[var_name] = np.concatenate(
            [output_labels[var_name] for output_labels in all_output_labels_list], axis=0)

    # Preprocess data
    input_features_scaled, output_labels_preprocessed, scaler = preprocess_data(
        input_features, combined_output_labels)

    # Define sequence length
    sequence_length = 100  # Adjust as needed

    # Create sequences
    X_sequences, y_sequences_dict = create_sequences(
        input_features_scaled, output_labels_preprocessed, sequence_length)

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train_dict, y_temp_dict = train_test_split(
        X_sequences, y_sequences_dict, test_size=0.3, random_state=42)

    X_val, X_test, y_val_dict, y_test_dict = train_test_split(
        X_temp, y_temp_dict, test_size=0.5, random_state=42)

    # Prepare decoder inputs (for training, we can use the same inputs as encoder inputs)
    X_train_dec = X_train
    X_val_dec = X_val
    X_test_dec = X_test

    num_features = X_train.shape[2]

    # Build and train model
    model = build_encoder_decoder_model(
        num_features, sequence_length, output_labels_preprocessed)
    train_model(model, X_train, X_train_dec, y_train_dict,
                X_val, X_val_dec, y_val_dict, epochs=20)

    # Evaluate model on test set
    test_loss = model.evaluate([X_test, X_test_dec], y_test_dict)
    print(f'Test Loss: {test_loss}')

    # Save the model
    model.save('encoder_decoder_model_multi_output.h5')

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
