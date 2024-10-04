import sys
import numpy as np
import scipy.io
import h5py
import keras
from keras import models, layers
from sklearn.preprocessing import StandardScaler
import tensorflow as tf  # or import torch if you prefer PyTorch


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
    """
    Load raw input data from the .mat file.
    """
    try:
        mat_contents = scipy.io.loadmat(
            input_filename, struct_as_record=False, squeeze_me=True)
        data = mat_contents.get('data', None)
        dataset = mat_contents.get('dataset', None)
        if data is None or dataset is None:
            raise ValueError(
                "The input .mat file does not contain 'data' and 'dataset' variables.")
        print("Input data loaded successfully.")
        return data, dataset
    except NotImplementedError:
        # Handle MATLAB v7.3 files
        print("Loading input .mat file using h5py.")
        with h5py.File(input_filename, 'r') as f:
            data = f.get('data', None)
            dataset = f.get('dataset', None)
            if data is None or dataset is None:
                raise ValueError(
                    "The input .mat file does not contain 'data' and 'dataset' variables.")
            print("Input data loaded successfully.")
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
        # Handle MATLAB v7.3 files
        print("Loading output .mat file using h5py.")
        with h5py.File(output_filename, 'r') as f:
            diss = f.get('diss', None)
            if diss is None:
                raise ValueError(
                    "The output .mat file does not contain 'diss' variable.")
            print("Output labels loaded successfully.")
            return diss


def extract_input_features(data, dataset, profile_num=0):
    """
    Extract input features from the raw data for a given profile number.
    """
    # Extract sampling frequencies from 'data'
    fs_fast = np.squeeze(data['fs_fast'])
    fs_slow = np.squeeze(data['fs_slow'])

    fs_fast = float(fs_fast)
    fs_slow = float(fs_slow)

    # Get the number of profiles
    # Adjusted to shape[0] based on possible data structure
    num_profiles = dataset['P_slow'].shape[0]
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

    # You can combine these variables into a feature array
    # For example, concatenate or stack them appropriately
    # For this example, let's assume we want to use sh1 and sh2 as input features
    # You might need to process them further, e.g., filtering, normalizing, etc.

    # Stack sh1 and sh2 to form input features
    sh = np.column_stack((sh1, sh2))  # Shape: (num_samples, 2)

    # Optionally, you can include other variables as features
    # For instance, combine Ax, Ay, T1_fast, etc.

    # Let's combine sh1, sh2, Ax, Ay, T1_fast into a feature array
    # Ensure that all variables have the same length
    min_length = min(len(sh1), len(sh2), len(Ax), len(Ay), len(T1_fast))
    sh1 = sh1[:min_length]
    sh2 = sh2[:min_length]
    Ax = Ax[:min_length]
    Ay = Ay[:min_length]
    T1_fast = T1_fast[:min_length]

    # Stack variables to form the input data
    input_features = np.column_stack((sh1, sh2, Ax, Ay, T1_fast))

    return input_features


def extract_output_labels(diss, profile_num=0):
    """
    Extract output labels from the dissipation data for a given profile number.
    """
    # Extract variables from the 'diss' structure
    e = diss.e  # Dissipation rate estimates
    # Ensure that e corresponds to the same profile_num
    # Assuming that the profiles are in the same order in the output file
    # and that 'e' is an array where each element corresponds to a profile

    if isinstance(e, np.ndarray) and e.ndim > 1:
        # Assuming e has shape (num_profiles, num_segments, num_probes)
        e_profile = e[profile_num]  # Shape: (num_segments, num_probes)
    else:
        raise ValueError("Unexpected format of 'e' in diss structure.")

    # You may need to flatten or reshape e_profile to match the input features
    # For simplicity, let's flatten it
    e_profile = e_profile.flatten()

    return e_profile


def align_input_output(input_features, output_labels):
    """
    Align input features and output labels.
    """
    # Ensure that the number of samples matches
    num_input_samples = input_features.shape[0]
    num_output_samples = len(output_labels)

    min_samples = min(num_input_samples, num_output_samples)

    # Trim both to the same length
    input_features = input_features[:min_samples]
    output_labels = output_labels[:min_samples]

    return input_features, output_labels


def preprocess_data(input_features, output_labels):
    """
    Preprocess input features and output labels.
    """
    # Handle NaNs in input features
    nan_indices = np.isnan(input_features).any(axis=1)
    input_features = input_features[~nan_indices]
    output_labels = output_labels[~nan_indices]

    # Handle NaNs or negative values in output labels
    valid_indices = ~np.isnan(output_labels) & (output_labels > 0)
    input_features = input_features[valid_indices]
    output_labels = output_labels[valid_indices]

    # Scale input features
    scaler = StandardScaler()
    input_features_scaled = scaler.fit_transform(input_features)

    # Optionally, log-transform the output labels if they span several orders of magnitude
    # Add small value to avoid log(0)
    output_labels_log = np.log10(output_labels + 1e-10)

    return input_features_scaled, output_labels_log, scaler


def prepare_data_for_model(input_features, output_labels, sequence_length=100):
    """
    Prepare data for the encoder-decoder model.
    """
    # Create sequences of the specified length
    num_samples = input_features.shape[0]
    num_sequences = num_samples // sequence_length

    input_sequences = []
    output_sequences = []

    for i in range(num_sequences):
        start_idx = i * sequence_length
        end_idx = start_idx + sequence_length
        input_seq = input_features[start_idx:end_idx]
        output_seq = output_labels[start_idx:end_idx]
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)

    # Shape: (num_sequences, sequence_length, num_features)
    input_sequences = np.array(input_sequences)
    # Shape: (num_sequences, sequence_length)
    output_sequences = np.array(output_sequences)

    # For an encoder-decoder model, the decoder typically produces a sequence
    # If you want to predict a single value per sequence, you can average or select a representative value
    # Here, we'll use the mean of the output sequence as the target
    output_sequences_mean = output_sequences.mean(
        axis=1)  # Shape: (num_sequences,)

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        input_sequences, output_sequences_mean, test_size=0.2, random_state=42)

    # Convert data to TensorFlow datasets
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(batch_size)

    return train_dataset, test_dataset


def build_encoder_decoder_model(input_shape):
    """
    Build an encoder-decoder model.
    """

    # Define the encoder
    encoder_inputs = layers.Input(shape=input_shape)
    encoder = layers.LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Define the decoder
    # Since we're predicting a single value per sequence, we can use a Dense layer
    decoder_dense = layers.Dense(1, activation='linear')
    decoder_outputs = decoder_dense(state_h)  # Use the encoder's hidden state

    # Build the model
    model = models.Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer='adam', loss='mse')

    print("Encoder-decoder model built successfully.")
    return model


def train_model(model, train_dataset, test_dataset, epochs=20):

    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
    print("Model training completed.")


def main():
    input_filename, output_filename = get_input_output_files()
    data, dataset = load_input_data(input_filename)
    diss = load_output_labels(output_filename)

    # Assume we're processing profile number 0
    profile_num = 0

    input_features = extract_input_features(data, dataset, profile_num)
    output_labels = extract_output_labels(diss, profile_num)

    # Align input and output data
    input_features_aligned, output_labels_aligned = align_input_output(
        input_features, output_labels)

    # Preprocess data
    input_features_scaled, output_labels_preprocessed, scaler = preprocess_data(
        input_features_aligned, output_labels_aligned)

    # Prepare data for model
    train_dataset, test_dataset = prepare_data_for_model(
        input_features_scaled, output_labels_preprocessed, sequence_length=100)

    # Build and train model
    input_shape = train_dataset.element_spec[0].shape[1:]  # Exclude batch size
    model = build_encoder_decoder_model(input_shape)
    train_model(model, train_dataset, test_dataset, epochs=20)

    # Save the model
    model.save('encoder_decoder_model.h5')

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
