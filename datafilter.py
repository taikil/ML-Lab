import numpy as np
import scipy.io
from scipy.signal import csd
import sys
import matplotlib.pyplot as plt
from nasmyth import nasmyth
import h5py
from clean_shear_spec import clean


def get_file():
    if len(sys.argv) == 2:
        filename = str(sys.argv[1])
        return filename
    else:
        print("Invalid Input; Add one url to the command")
        print("Exiting Program...")
        sys.exit()


def load_mat_v73(filename):
    """
    Load a MATLAB v7.3 .mat file using h5py.

    Parameters:
        filename (str): Path to the .mat file.

    Returns:
        data (dict): Dictionary containing the data.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in f.keys():
            data[key] = load_hdf5_group(f[key])
    return data


def load_hdf5_group(group):
    """
    Recursively load an HDF5 group into a dictionary.

    Parameters:
        group (h5py.Group or h5py.Dataset): The group or dataset to load.

    Returns:
        data (dict or ndarray): Loaded data.
    """
    if isinstance(group, h5py.Dataset):
        return group[()]
    elif isinstance(group, h5py.Group):
        data = {}
        for key in group.keys():
            data[key] = load_hdf5_group(group[key])
        return data
    else:
        raise TypeError("Unsupported HDF5 object type.")


def inspect_mat_data(mat_data):
    """
    Recursively print the structure of the loaded .mat data.

    Parameters:
        mat_data (dict): The loaded .mat data.
    """
    def print_dict(d, indent=0):
        for key, value in d.items():
            print('  ' * indent + f"Key: {key}")
            if isinstance(value, dict):
                print_dict(value, indent + 1)
            else:
                print('  ' * (indent + 1) +
                      f"Type: {type(value)}, Shape: {value.shape}")

    print_dict(mat_data)


def visc35(T):
    # Kinematic viscosity of seawater at 35 PSU (salinity)
    # T in degrees Celsius
    nu = 1e-6 * (17.91 - 0.5381 * T + 0.00694 * T**2)
    return nu


def inertial_subrange(K, shear_spectrum, e_initial, nu, K_limit):
    # (Implementation as provided earlier)
    # [Include the inertial_subrange function code here]
    pass  # Replace with the actual function code


def get_dissipation(SH, A, P, T, speed, fs, fft_length=1024, diss_length=3072, overlap=1536):
    # Input validation and setup
    if diss_length < 2 * fft_length:
        raise ValueError('diss_length must be greater than 2 * fft_length.')
    if len(SH) != len(A) or len(SH) != len(T) or len(SH) != len(P):
        raise ValueError('SH, A, T, and P must have the same length.')
    if not np.isscalar(speed) and len(speed) != len(SH):
        raise ValueError(
            'speed must be a scalar or have the same length as SH.')
    if diss_length > len(SH):
        raise ValueError('diss_length cannot be longer than the length of SH.')

    # Adjust overlap if necessary
    if overlap >= diss_length:
        overlap = diss_length // 2
    if overlap < 0:
        overlap = 0

    # Calculate the number of dissipation estimates
    step = diss_length - overlap
    num_estimates = 1 + (len(SH) - diss_length) // step
    f_length = fft_length // 2 + 1

    # Pre-allocate arrays
    diss_e = np.zeros((num_estimates, SH.shape[1]))
    diss_K_max = np.zeros_like(diss_e)
    diss_Nasmyth_spec = np.zeros((num_estimates, SH.shape[1], f_length))
    diss_F = np.zeros((num_estimates, f_length))
    diss_K = np.zeros_like(diss_F)
    diss_speed = np.zeros(num_estimates)
    diss_nu = np.zeros(num_estimates)
    diss_P = np.zeros(num_estimates)
    diss_T = np.zeros(num_estimates)

    a = 1.0774e9  # From Lueck's model for e/e_10
    x_isr = 0.02  # Adjusted non-dimensional wavenumber limit

    index = 0
    select_start = 0
    select_end = diss_length

    while select_end <= len(SH):
        select = slice(select_start, select_end)

        # Extract data segment
        U_segment = SH[select]
        A_segment = A[select]
        P_segment = P[select]
        T_segment = T[select]
        speed_segment = speed if np.isscalar(speed) else speed[select]

        # Clean shear spectra
        clean_UU, AA, UU, UA, F = clean(
            A_segment, U_segment, fft_length, fs)

        # Extract the cleaned auto-spectra for each shear probe
        P_sh_clean = np.zeros((SH.shape[1], f_length))
        for i in range(SH.shape[1]):
            P_sh_clean[i, :] = np.real(clean_UU[i, i, :])

        # Convert frequency spectra to wavenumber spectra
        W = np.mean(np.abs(speed_segment))
        K = F / W
        K_AA = 0.9 * (fs / 2) / W  # Anti-aliasing wavenumber limit

        # Wavenumber correction
        correction = np.ones_like(P_sh_clean)
        correction_indices = K <= 150
        correction[:, correction_indices] += (K[correction_indices] / 48) ** 2
        P_sh_clean *= correction

        # Kinematic viscosity
        mean_T = np.mean(T_segment)
        nu = visc35(mean_T)

        # Initialize variables
        e = np.zeros(SH.shape[1])
        K_max = np.zeros(SH.shape[1])
        Nasmyth_spectrum = np.zeros((SH.shape[1], len(K)))

        for col in range(SH.shape[1]):
            shear_spectrum = P_sh_clean[col, :]

            # Estimate e_10
            K_range = K <= 10
            e_10 = 7.5 * nu * np.trapz(shear_spectrum[K_range], K[K_range])
            e_1 = e_10 * np.sqrt(1 + a * e_10)

            # Determine method based on e_1
            if e_1 < 0.09:  # Use variance method
                # Implement variance method calculations (simplified)
                e_4 = e_1  # Placeholder for actual calculations
            else:  # Use inertial subrange fitting
                K_limit = min(K_AA, 150)
                e_4, K_end = inertial_subrange(
                    K, shear_spectrum, e_1, nu, K_limit)
                e[col] = e_4
                K_max[col] = K_end
                Nasmyth_spectrum[col, :] = nasmyth(e_4, nu, K)

        # Store results
        diss_e[index, :] = e
        diss_Nasmyth_spec[index, :, :] = Nasmyth_spectrum
        diss_K_max[index, :] = K_max
        diss_F[index, :] = F
        diss_K[index, :] = K
        diss_speed[index] = W
        diss_nu[index] = nu
        diss_T[index] = mean_T
        diss_P[index] = np.mean(P_segment)

        # Update indices
        index += 1
        select_start += step
        select_end = select_start + diss_length

    # Compile results into a dictionary
    diss = {
        'e': diss_e,
        'K_max': diss_K_max,
        'Nasmyth_spec': diss_Nasmyth_spec,
        'F': diss_F,
        'K': diss_K,
        'speed': diss_speed,
        'nu': diss_nu,
        'T': diss_T,
        'P': diss_P
    }

    return diss


def main():
    # Load the .mat file
    filename = get_file()
    mat_data = load_mat_v73(filename)
    inspect_mat_data(mat_data)
    data = mat_data['data']

    # Extract variables
    SH1 = data.sh1
    SH2 = data.sh2
    SH = np.column_stack((SH1, SH2))

    Ax = data.Ax
    Ay = data.Ay
    A = np.column_stack((Ax, Ay))

    P = data.P
    T = data.T1
    speed = data.speed_fast  # If not available, calculate from pressure data
    fs = data.fs_fast

    # Ensure all arrays have the same length
    min_length = min(len(SH), len(A), len(T), len(P), len(speed))
    SH = SH[:min_length]
    A = A[:min_length]
    T = T[:min_length]
    P = P[:min_length]
    speed = speed[:min_length]

    # Call the get_dissipation function
    diss = get_dissipation(SH, A, P, T, speed, fs)

    # Plot results
    plt.figure()
    for i in range(diss['e'].shape[1]):
        plt.plot(diss['e'][:, i], diss['P'], label=f'Shear Probe {i+1}')
    plt.gca().invert_yaxis()
    plt.xlabel('Dissipation Rate (W/kg)')
    plt.ylabel('Pressure (dbar)')
    plt.title('Dissipation Rate Profile')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
