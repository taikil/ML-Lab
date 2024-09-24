import sys
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import h5py
import keras
from keras import Input
import matplotlib.pyplot as plt
from scipy.stats import zscore


def get_file():
    if len(sys.argv) == 2:
        url = str(sys.argv[1])
        return url
    else:
        print("Invalid Input; Add one url to the command")
        print("Exiting Program...")
        sys.exit()


# data_analysis.py
# Global Variables
FILENAME = ''  # Replace with the path to your .mat file
THRESHOLD_PERCENTILE = 95   # Percentile for anomaly detection threshold


def load_mat_file(filename):
    """
    Load data from a .mat file and return relevant datasets.
    """
    """
       Load data from a .mat file and return relevant datasets.
       """
    mat = scipy.io.loadmat(filename)

    # Access the 'diss' structure
    diss = mat['diss']

    # Extract datasets
    diss_e = diss['e'][0, 0]
    diss_K = diss['K'][0, 0]
    diss_sh_clean = diss['sh_clean'][0, 0]
    diss_F = diss['F'][0, 0]
    diss_flagood = diss['flagood'][0, 0].flatten()

    print("Data loaded successfully.")
    return diss_e, diss_K, diss_sh_clean, diss_F, diss_flagood


def preprocess_data(diss_e, diss_sh_clean):
    """
    Handle missing values and scale the data.
    """
    # Remove NaNs
    nan_indices = np.isnan(diss_e).any(axis=1)
    diss_e = diss_e[~nan_indices]
    diss_sh_clean = diss_sh_clean[~nan_indices]

    # Flatten the shear spectra
    num_samples, num_probes, _, num_freqs = diss_sh_clean.shape
    diss_sh_clean_reshaped = diss_sh_clean.reshape(num_samples, -1)

    # Scale the data
    scaler = StandardScaler()
    diss_sh_clean_scaled = scaler.fit_transform(diss_sh_clean_reshaped)

    print("Data preprocessing completed.")
    return diss_e, diss_sh_clean_scaled, scaler


def extract_features(diss_sh_clean_scaled):
    """
    Extract features from the shear spectra data.
    """
    # Calculate statistical features
    shear_mean = np.mean(diss_sh_clean_scaled, axis=1)
    shear_var = np.var(diss_sh_clean_scaled, axis=1)
    shear_skew = np.mean(
        (diss_sh_clean_scaled - shear_mean[:, np.newaxis])**3, axis=1) / (shear_var ** 1.5)
    shear_kurtosis = np.mean(
        (diss_sh_clean_scaled - shear_mean[:, np.newaxis])**4, axis=1) / (shear_var ** 2)

    # Combine features
    X = np.column_stack((shear_mean, shear_var, shear_skew, shear_kurtosis))

    print("Feature extraction completed.")
    return X


def build_autoencoder(input_dim):
    """
    Define and compile the autoencoder model.
    """
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoder = keras.layers.Dense(
        encoding_dim, activation='relu')(input_layer)
    encoder = keras.layers.Dense(16, activation='relu')(encoder)
    decoder = keras.layers.Dense(encoding_dim, activation='relu')(encoder)
    decoder = keras.layers.Dense(input_dim, activation='linear')(decoder)

    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    print("Autoencoder model built.")
    return autoencoder


def train_autoencoder(autoencoder, X_train, X_val):
    """
    Train the autoencoder model.
    """
    history = autoencoder.fit(X_train, X_train,
                              epochs=50,
                              batch_size=32,
                              shuffle=True,
                              validation_data=(X_val, X_val))

    print("Autoencoder training completed.")
    return history


def evaluate_model(autoencoder, X_train, X_val, THRESHOLD_PERCENTILE):
    """
    Evaluate the model and determine the anomaly detection threshold.
    """
    # Reconstruction errors on training data
    reconstructions = autoencoder.predict(X_train)
    train_loss = np.mean(np.square(reconstructions - X_train), axis=1)

    # Set threshold
    threshold = np.percentile(train_loss, THRESHOLD_PERCENTILE)
    print(f"Reconstruction error threshold: {threshold}")

    # Reconstruction errors on validation data
    val_reconstructions = autoencoder.predict(X_val)
    val_loss = np.mean(np.square(val_reconstructions - X_val), axis=1)

    return threshold, val_loss


def filter_anomalies(autoencoder, X, threshold):
    """
    Use the autoencoder to filter out anomalies in the dataset.
    """
    reconstructions = autoencoder.predict(X)
    losses = np.mean(np.square(reconstructions - X), axis=1)

    # Identify normal data points
    normal_indices = np.where(losses <= threshold)[0]
    anomaly_indices = np.where(losses > threshold)[0]

    print(f"Number of normal data points: {len(normal_indices)}")
    print(f"Number of anomalies detected: {len(anomaly_indices)}")

    return normal_indices, anomaly_indices


def main():
    # Step 1: Load and Explore the Data
    FILENAME = get_file()
    diss_e, diss_K, diss_sh_clean, diss_F, diss_flagood = load_mat_file(
        FILENAME)

    # Step 2: Data Preprocessing
    diss_e, diss_sh_clean_scaled, scaler = preprocess_data(
        diss_e, diss_sh_clean)

    # Step 3: Feature Engineering
    X = extract_features(diss_sh_clean_scaled)

    # Step 4: Identify Outliers in diss_e (Optional, using z-scores)
    diss_e_zscores = zscore(diss_e, axis=0)
    threshold_z = 3  # Adjust as needed
    outlier_indices = np.where(np.abs(diss_e_zscores) > threshold_z)[0]
    y = np.ones(len(diss_e))
    y[outlier_indices] = -1  # Label outliers as -1

    # Step 5: Build the Autoencoder Model
    input_dim = X.shape[1]
    autoencoder = build_autoencoder(input_dim)

    # Step 6: Training the Model
    # Use only normal data for training
    normal_data = X[y == 1]
    X_train, X_val = train_test_split(
        normal_data, test_size=0.2, random_state=42)
    history = train_autoencoder(autoencoder, X_train, X_val)

    # Step 7: Evaluating the Model
    threshold, val_loss = evaluate_model(
        autoencoder, X_train, X_val, THRESHOLD_PERCENTILE)

    # Step 8: Filter Anomalies in the Entire Dataset
    normal_indices, anomaly_indices = filter_anomalies(
        autoencoder, X, threshold)

    # Filter dissipation rates
    filtered_diss_e = diss_e[normal_indices]
    filtered_diss_e_anomalies = diss_e[anomaly_indices]

    # Optional: Visualize the filtered dissipation rates
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_diss_e, label='Filtered Dissipation Rate')
    plt.xlabel('Sample Index')
    plt.ylabel('Dissipation Rate (W/kg)')
    plt.title('Filtered Dissipation Rate Over Samples')
    plt.legend()
    plt.show()

    print("Data filtering completed.")


if __name__ == "__main__":
    main()
