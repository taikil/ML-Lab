import sys
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def get_file():
    if len(sys.argv) == 2:
        filename = str(sys.argv[1])
        return filename
    else:
        print("Invalid Input; Provide the .mat filename as a command-line argument.")
        print("Exiting Program...")
        sys.exit()


def load_data(filename):
    """
    Load data from the .mat file and extract relevant variables.
    """
    mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    diss = mat['diss']

    # Extract variables
    e = diss.e  # [number_of_segments, number_of_probes]
    K_max = diss.K_max  # [number_of_segments, number_of_probes]
    # [number_of_segments, number_of_probes, number_of_wavenumbers]
    Nasmyth_spec = diss.Nasmyth_spec
    # [number_of_segments, number_of_probes, number_of_probes, number_of_wavenumbers]
    sh_clean = diss.sh_clean
    F = diss.F  # [number_of_segments, number_of_frequencies]
    K = diss.K  # [number_of_segments, number_of_frequencies]
    speed = diss.speed  # [number_of_segments, 1]
    nu = diss.nu  # [number_of_segments, 1]
    P = diss.P  # [number_of_segments, 1]
    T = diss.T  # [number_of_segments, 1]
    N2 = diss.N2  # [number_of_segments, number_of_probes]
    Krho = diss.Krho  # [number_of_segments, number_of_probes]
    flagood = diss.flagood  # [number_of_segments, number_of_probes]

    print("Data loaded successfully.")
    return e, K_max, Nasmyth_spec, sh_clean, F, K, speed, nu, P, T, N2, Krho, flagood


def preprocess_data(e, K_max, Nasmyth_spec, sh_clean, F, K, speed, nu, P, T, N2, Krho, flagood):
    # Handle NaNs across all probes
    nan_indices = np.isnan(e).any(axis=1)

    # Filter out rows with NaNs
    e = e[~nan_indices]
    K_max = K_max[~nan_indices]
    Nasmyth_spec = Nasmyth_spec[~nan_indices]
    sh_clean = sh_clean[~nan_indices]
    F = F[~nan_indices]
    K = K[~nan_indices]
    speed = speed[~nan_indices]
    nu = nu[~nan_indices]
    P = P[~nan_indices]
    T = T[~nan_indices]
    N2 = N2[~nan_indices]
    Krho = Krho[~nan_indices]
    flagood = flagood[~nan_indices]

    # Number of segments and probes
    num_segments, num_probes = e.shape

    # Initialize lists to collect features for all probes
    features_list = []
    e_list = []

    for probe_idx in range(num_probes):
        # Extract features for each probe
        e_probe = e[:, probe_idx]
        K_max_probe = K_max[:, probe_idx]
        N2_probe = N2[:, probe_idx]
        Krho_probe = Krho[:, probe_idx]
        flagood_probe = flagood[:, probe_idx]

        # Handle NaNs in e for this probe
        valid_indices = ~np.isnan(e_probe)
        e_probe = e_probe[valid_indices]
        K_max_probe = K_max_probe[valid_indices]
        N2_probe = N2_probe[valid_indices]
        Krho_probe = Krho_probe[valid_indices]
        flagood_probe = flagood_probe[valid_indices]
        speed_probe = speed[valid_indices]
        nu_probe = nu[valid_indices]
        P_probe = P[valid_indices]
        T_probe = T[valid_indices]
        Nasmyth_spec_probe = Nasmyth_spec[valid_indices, probe_idx, :]
        sh_clean_probe = sh_clean[valid_indices, probe_idx, probe_idx, :]

        # Extract statistical features from Nasmyth_spec and sh_clean
        Nasmyth_mean = np.mean(Nasmyth_spec_probe, axis=1)
        Nasmyth_var = np.var(Nasmyth_spec_probe, axis=1)
        sh_mean = np.mean(sh_clean_probe, axis=1)
        sh_var = np.var(sh_clean_probe, axis=1)

        # Construct feature matrix for this probe
        X_probe = np.column_stack((
            K_max_probe,
            speed_probe.flatten(),
            nu_probe.flatten(),
            P_probe.flatten(),
            T_probe.flatten(),
            N2_probe,
            Krho_probe,
            flagood_probe,
            Nasmyth_mean,
            Nasmyth_var,
            sh_mean,
            sh_var
        ))

        # Append features and target to the lists
        features_list.append(X_probe)
        e_list.append(e_probe)

    # Concatenate features and targets from all probes
    X = np.vstack(features_list)
    y = np.concatenate(e_list)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data preprocessing completed.")
    return X_scaled, y, scaler


def train_regression_model(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse:.5e}")
    print(f"Model R^2 Score: {r2:.5f}")

    return model, X_test, y_test, y_pred


def plot_results(y_test, y_pred):
    """
    Plot the actual vs. predicted dissipation rates.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [
             y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Dissipation Rate (e) [W/kg]')
    plt.ylabel('Predicted Dissipation Rate (e) [W/kg]')
    plt.title('Actual vs. Predicted Dissipation Rate')
    plt.grid(True)
    plt.show()


def main():
    FILENAME = get_file()
    e, K_max, Nasmyth_spec, sh_clean, F, K, speed, nu, P, T, N2, Krho, flagood = load_data(
        FILENAME)

    X_scaled, y, scaler = preprocess_data(
        e, K_max, Nasmyth_spec, sh_clean, F, K, speed, nu, P, T, N2, Krho, flagood)

    model, X_test, y_test, y_pred = train_regression_model(X_scaled, y)

    plot_results(y_test, y_pred)

    print("Processing completed.")


if __name__ == "__main__":
    main()
