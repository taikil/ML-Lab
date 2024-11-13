import sys
import numpy as np
import scipy.io
from keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import hdf5storage
from predict_dissipation import *


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


# def extract_output_labels(diss):
#     """
#     Extract output labels from the dissipation data.
#     """
#     # List of variables to extract from 'diss'
#     output_variables = [
#         'e', 'K_max', 'warning', 'method', 'Nasmyth_spec', 'sh',
#         'sh_clean', 'AA', 'UA', 'F', 'K', 'speed', 'nu', 'P', 'T', 'flagood'
#     ]

#     # Dictionary to store the extracted variables
#     output_labels = {}

#     for var_name in output_variables:
#         try:
#             var_data = getattr(diss, var_name)
#             # Handle different dimensions
#             if isinstance(var_data, np.ndarray):
#                 if var_data.ndim == 2:
#                     # For 2D arrays (e.g., 101x2), reshape to 1D
#                     var_profile = var_data.reshape(-1)
#                 elif var_data.ndim > 2:
#                     # For higher-dimensional arrays, reshape accordingly
#                     var_profile = var_data.reshape(
#                         var_data.shape[0] * var_data.shape[1], -1)
#                 else:
#                     var_profile = var_data
#             else:
#                 var_profile = var_data  # Scalar or non-array

#             output_labels[var_name] = var_profile
#         except KeyError:
#             raise KeyError(
#                 f"Variable '{var_name}' not found in 'diss' structure.")
#         except Exception as e:
#             raise Exception(f"Error extracting variable '{var_name}': {e}")

#     return output_labels


def extract_output_labels(diss):
    """
    Extract output labels from the dissipation data for a given profile number.
    """
    # List of variables to extract from 'diss'
    output_variables = ['e', 'K_max', 'Nasmyth_spec']

    # Dictionary to store the extracted variables
    output_labels = {}

    for var_name in output_variables:
        try:
            var_data = getattr(diss, var_name)
            if isinstance(var_data, np.ndarray) and var_data.ndim > 1:
                print(f"Shape: {var_name} : {var_data.shape}")
                var_data = var_data.flatten()
            output_labels[var_name] = var_data
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
    print(f"Input sample len: {num_input_samples}")
    print(f"input_features len: {len(input_features)}")
    print(f"output features len: {output_labels['e']}")
    min_samples = num_input_samples

    # Determine the minimum length across all output labels
    for var_name, var_data in output_labels.items():
        if isinstance(var_data, np.ndarray):
            print(f"var name! {var_name}")
            print(f"var data! {var_data}")
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


def create_windowed_dataset(features, labels, window_size, overlap):
    step_size = window_size - overlap
    num_samples = len(features)
    X = []
    y = []
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        X_window = features[start:end]
        y_window = labels[start:end]
        X.append(X_window)
        y.append(np.mean(y_window))  # Use mean dissipation rate in the window
    return np.array(X), np.array(y)


def train_model(model, X_train_enc, X_train_dec, y_train_dict, X_val_enc, X_val_dec, y_val_dict, epochs=20):
    """
    Train the encoder-decoder model.
    """
    model.fit([X_train_enc, X_train_dec], y_train_dict,
              epochs=epochs,
              batch_size=32,
              validation_data=([X_val_enc, X_val_dec], y_val_dict))
    print("Model training completed.")


def plot(model, history, X_test, y_test):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

    y_pred = model.predict(X_test)

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Dissipation Rate')
    plt.ylabel('Predicted Dissipation Rate')
    plt.title('Actual vs. Predicted Dissipation Rate')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.show()


def main():
    # Assuming fs_fast is the sampling frequency from your data
    params = {
        'fft_length': 4.0,      # FFT length in seconds (from MATLAB code)
        'diss_length': 8.0,     # Window length in seconds
        'overlap': 4.0,         # Overlap length in seconds
        'profile_num': 0
    }

    input_filename, output_filename = get_input_output_files()
    data, dataset = load_input_data(input_filename)
    diss = load_output_data(output_filename)

    # Convert to samples
    fs_fast = float(np.squeeze(data['fs_fast']))
    params['fs_fast'] = fs_fast

    fft_length_samples = int(params['fft_length'] * params['fs_fast'])
    diss_length_samples = int(params['diss_length'] * params['fs_fast'])
    overlap_samples = int(params['overlap'] * params['fs_fast'])
    num_profiles = len(dataset['P_slow'])

    input_features = extract_input_features(
        data, dataset, params['profile_num'])
    print(input_features)
    output_labels = extract_output_labels(diss)

    input_features_aligned, output_labels_aligned = align_input_output(
        input_features, output_labels)

    # Preprocess data
    input_features_scaled, output_labels_preprocessed, scaler = preprocess_data(
        input_features_aligned, output_labels_aligned)

    #  # Extract the dissipation rate 'e' for further processing
    output_labels_e = output_labels_preprocessed['e']

    # Create windowed dataset
    X_windows, y_windows = create_windowed_dataset(
        input_features_scaled, output_labels_e, diss_length_samples, overlap_samples)

    # Expected: (num_windows, window_size, num_features)
    print("X_windows shape:", X_windows.shape)
    print("y_windows shape:", y_windows.shape)  # Expected: (num_windows,)

    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_windows, y_windows, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42)

    # Define model
    model = models.Sequential()
    model.add(layers.Conv1D(64, 3, activation='relu',
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    plot(model, history, X_test, y_test)

    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Save model
    model.save('dissipation_rate_cnn_model.h5')


if __name__ == "__main__":
    main()
