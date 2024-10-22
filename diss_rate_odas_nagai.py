import numpy as np
from nasmyth import nasmyth
from clean_shear_spec import clean


def visc35(T):
    # Polynomial coefficients (from highest degree to constant term)
    pol = [
        -1.131311019739306e-11,   # Coefficient for t³
        1.199552027472192e-09,   # Coefficient for t²
        -5.864346822839289e-08,   # Coefficient for t¹
        1.828297985908266e-06    # Constant term
    ]
    # Evaluate the polynomial at temperature t
    v = np.polyval(pol, T)
    return v


def fit_in_range(value, range_vals, default=None):
    if value is not None and (np.size(value) > 0):
        result = np.atleast_1d(value)[0]
    else:
        result = np.atleast_1d(default)[0]

    # Fit result into the specified range
    if result < range_vals[0]:
        result = range_vals[0]
    if result > range_vals[1]:
        result = range_vals[1]
    return result


def inertial_subrange(K, shear_spectrum, e, nu, K_limit):
    """
    Find the best fit to the inertial subrange by adjusting the rate of dissipation.

    Parameters:
    - K: array_like
        Wavenumbers, including zero.
    - shear_spectrum: array_like
        Wavenumber spectrum of shear.
    - e: float
        Starting value of the rate of dissipation, usually derived from the e/e_10 model of Lueck.
    - nu: float
        Kinematic viscosity.
    - K_limit: float
        Wavenumber limits other than the inertial subrange (e.g., 150 cpm or K_AA).

    Returns:
    - e: float
        Adjusted rate of dissipation.
    - K_max: float
        Maximum wavenumber used in the fitting.
    """
    x_isr = 0.01  # Non-dimensional limit of the inertial subrange
    x_isr *= 2    # Pushing this up a little

    # Calculate the limit of the inertial subrange
    K_isr_limit = x_isr * (e / nu**3) ** 0.25
    K_fit_limit = min(K_isr_limit, K_limit)

    # Determine the fitting range
    fit_range = K <= K_fit_limit
    fit_indices = np.where(fit_range)[0]

    if len(fit_indices) == 0:
        raise ValueError(
            "No data points in the inertial subrange fitting range.")

    K_max = K[fit_indices[-1]]

    # Less than 9 points in inertial subrange (assuming K[0] = 0)
    if K_max < 10 * K[1]:
        K_max = K[1]       # Add a warning or error flag here if needed
        return e, K_max

    # Perform iterative fitting
    for _ in range(3):
        Nasmyth_values = nasmyth(e, nu, K[fit_range])
        ratio = shear_spectrum[fit_range][1:] / Nasmyth_values[1:]
        fit_error = np.mean(np.log10(ratio))
        e *= 10 ** (1.5 * fit_error)

    # Remove outliers (flyers)
    Nasmyth_values = nasmyth(e, nu, K[fit_range])
    ratio = shear_spectrum[fit_range][1:] / Nasmyth_values[1:]
    fit_error = np.log10(ratio)
    flyers_index = np.where(np.abs(fit_error) > 0.5)[0]
    if flyers_index.size > 0:
        # Remove up to 20% of points
        bad_limit = int(np.ceil(0.2 * len(fit_error)))
        if len(flyers_index) > bad_limit:
            flyers_index = flyers_index[:bad_limit]
        # Remove the flyers from the fit range
        fit_indices = fit_indices[1:]  # Exclude K[0]
        fit_indices = np.delete(fit_indices, flyers_index)
        fit_range = np.zeros_like(K, dtype=bool)
        fit_range[fit_indices] = True

    K_max = K[fit_range][-1]

    # Refit after removing outliers
    for _ in range(2):
        Nasmyth_values = nasmyth(e, nu, K[fit_range])
        ratio = shear_spectrum[fit_range][1:] / Nasmyth_values[1:]
        fit_error = np.mean(np.log10(ratio))
        e *= 10 ** (1.5 * fit_error)

    return e, K_max


def get_diss_odas_nagai4gui2024(SH, A, fft_length, diss_length, overlap, fs,
                                speed, T, N2, P, fit_order=5, f_AA=98):
    """
    Calculate dissipation rates over an entire profile.
    """

    # Check inputs
    if diss_length < 2 * fft_length:
        raise ValueError(
            'Invalid size for diss_length - must be greater than 2 * fft_length.')

    if SH.shape[0] != A.shape[0] or SH.shape[0] != T.shape[0] or SH.shape[0] != P.shape[0]:
        raise ValueError(
            f"Same number of rows required for SH, A, T, P... SH: {SH.shape[0]}, A: {A.shape[0]}, T: {T.shape[0]}, P: {P.shape[0]}")

    if not np.isscalar(speed) and len(speed) != SH.shape[0]:
        raise ValueError(
            f'Speed vector must have the same number of rows as shear, speed: {len(speed)}')

    if diss_length > SH.shape[0]:
        raise ValueError(
            'Diss_length cannot be longer than the length of the shear vectors')

    f_AA *= 0.9  # constrain limit to 90% of F_AA

    select = np.arange(diss_length)

    # Calculate the number of dissipation estimates
    if overlap >= diss_length:
        overlap = diss_length / 2
    if overlap < 0:
        overlap = 0
    number_of_rows = 1 + \
        int(np.floor((SH.shape[0] - diss_length) / (diss_length - overlap)))
    F_length = 1 + int(np.floor(fft_length / 2))

    # Pre-allocate matrices
    diss = {}
    num_probes = SH.shape[1]
    diss['e'] = np.zeros((number_of_rows, num_probes))
    diss['K_max'] = np.zeros_like(diss['e'])
    diss['method'] = np.zeros_like(diss['e'])
    diss['Nasmyth_spec'] = np.zeros((number_of_rows, num_probes, F_length))
    diss['sh'] = np.zeros((number_of_rows, num_probes, num_probes, F_length))
    diss['sh_clean'] = np.zeros_like(diss['sh'])
    diss['AA'] = np.zeros((number_of_rows, A.shape[1], A.shape[1], F_length))
    diss['UA'] = np.zeros((number_of_rows, num_probes, A.shape[1], F_length))
    diss['F'] = np.zeros((number_of_rows, F_length))
    diss['K'] = np.zeros_like(diss['F'])
    diss['speed'] = np.zeros((number_of_rows, 1))
    diss['nu'] = np.zeros_like(diss['speed'])
    diss['P'] = np.zeros_like(diss['speed'])
    diss['T'] = np.zeros_like(diss['speed'])
    diss['N2'] = np.zeros((number_of_rows, num_probes))
    diss['Krho'] = np.zeros_like(diss['N2'])
    diss['flagood'] = np.zeros_like(diss['e'])

    a = 1.0774e9  # From Lueck's model for e/e_10
    x_isr = 0.02  # Adjusted non-dimensional wavenumber
    x_95 = 0.1205  # To be verified

    index = 0
    while select[-1] < SH.shape[0]:
        # Call clean_shear_spec
        P_sh_clean, AA, P_sh, UA, F = clean(
            A[select, :], SH[select, :], fft_length, fs)
        # P_sh_clean and P_sh have shape [num_probes, num_probes, F_length]

        # Convert frequency spectra to wavenumber spectra
        W = np.mean(np.abs(speed[select]))
        K = F / W
        K_AA = f_AA / W
        num_probes = SH.shape[1]
        correction = np.ones((num_probes, num_probes, len(K)))

        K_broadcast = K[np.newaxis, np.newaxis, :]  # Shape: (1, 1, len(K))
        # Shape: (num_probes, num_probes, len(K))
        K_broadcast = np.broadcast_to(K_broadcast, correction.shape)

        correction_indices = K_broadcast <= 150
        correction[correction_indices] = 1 + \
            (K_broadcast[correction_indices] / 48) ** 2

        P_sh_clean *= W * correction
        P_sh *= W * correction

        e = np.zeros(num_probes)
        K_max = np.zeros(num_probes)
        method = np.zeros(num_probes)
        flagood = np.zeros(num_probes)

        mean_T = np.mean(T[select])
        nu = visc35(mean_T)

        for column_index in range(num_probes):
            # Get auto-spectrum
            if num_probes == 1:
                shear_spectrum = P_sh_clean.squeeze()
            else:
                shear_spectrum = P_sh_clean[column_index, column_index, :]

            K_range = K <= 10
            e_10 = 7.5 * nu * np.trapz(shear_spectrum[K_range], K[K_range])
            e_1 = e_10 * np.sqrt(1 + a * e_10)

            if e_1 < 9e-2:
                # Use variance method
                e_2 = e_1
                K_95 = x_95 * (e_2 / nu ** 3) ** 0.25
                K_95 = max(K_95, np.finfo(float).eps)  # Non-zero
                K_limit = fit_in_range(min(K_AA, K_95), [0, 150])
                valid_shear = K <= K_limit
                Index_limit = np.sum(valid_shear)
                y = np.log10(shear_spectrum[1:Index_limit])
                x = np.log10(K[1:Index_limit])
                fit_order = int(fit_in_range(fit_order, [3, 8]))
                if Index_limit > fit_order + 2:
                    p = np.polyfit(x, y, fit_order)
                    pd1 = np.polyder(p)
                    pr1 = np.roots(pd1)
                    pr1 = pr1[np.isreal(pr1)]
                    # minima only
                    pr1 = pr1[np.polyval(np.polyder(pd1), pr1) > 0]
                    pr1 = pr1[pr1 >= np.log10(10)]
                    if pr1.size == 0:
                        pr1 = np.log10(K_95)
                    else:
                        pr1 = pr1[0]
                else:
                    pr1 = np.log10(K_95)
                K_limit = fit_in_range(min([pr1, np.log10(K_95), np.log10(K_AA)]),
                                       [np.log10(10), np.log10(150)],
                                       np.log10(150))
                Range = K <= 10 ** K_limit
                e_3 = 7.5 * nu * np.trapz(shear_spectrum[Range], K[Range])

                x_limit = K[Range][-1] * (nu ** 3 / e_3) ** 0.25
                x_limit **= (4 / 3)

                variance_resolved = np.tanh(
                    48 * x_limit) - 2.9 * x_limit * np.exp(-22.3 * x_limit)

                e_new = e_3 / variance_resolved
                if not np.isfinite(e_new):
                    print("Warning: e_new is invalid.")
                    e[column_index] = np.nan
                    method[column_index] = np.nan
                    flagi = 0
                    continue

                # Iterative correction
                max_iterations = 100
                iteration_count = 0
                while True:
                    iteration_count += 1
                    if iteration_count >= max_iterations:
                        print(
                            "Warning: Maximum iterations reached without convergence.")
                        e[column_index] = np.nan
                        method[column_index] = np.nan
                        flagi = 0
                        break
                    x_limit = K[Range][-1] * (nu ** 3 / e_new) ** 0.25
                    x_limit **= (4 / 3)
                    variance_resolved = np.tanh(
                        48 * x_limit) - 2.9 * x_limit * np.exp(-22.3 * x_limit)
                    e_old = e_new
                    e_new = e_3 / variance_resolved
                    if e_new / e_old < 1.02:
                        e_3 = e_new
                        e_3 = float(e_3)
                        break

               # Correct for missing variance at low end
                phi, k = nasmyth(e_3, nu, K[1:3])
                # phi, k = nasmyth(e_3, nu, K[1])

                e_4 = e_3 + 0.25 * 7.5 * nu * K[1] * phi[0]
                if isinstance(e_4, np.ndarray):
                    e_4 = e_4[0]
                e_4 = float(e_4)
                if e_4 / e_3 > 1.1:
                    e_new = e_4 / variance_resolved
                    while True:
                        x_limit = K[Range][-1] * (nu ** 3 / e_new) ** 0.25
                        x_limit **= (4 / 3)
                        variance_resolved = np.tanh(
                            48 * x_limit) - 2.9 * x_limit * np.exp(-22.3 * x_limit)
                        e_old = e_new
                        e_new = e_4 / variance_resolved
                        if e_new / e_old < 1.02:
                            e_4 = e_new
                            break

                K_max[column_index] = K[Range][-1]
                e[column_index] = e_4
                method[column_index] = 0
                flagi = 1
            else:
                # Use inertial subrange method
                K_limit = min(K_AA, 150)
                e_4, K_end = inertial_subrange(
                    K, shear_spectrum, e_1, nu, K_limit)
                K_max[column_index] = K_end
                e[column_index] = e_4
                method[column_index] = 1
                flagi = 1

            # Compute Krho
            Krho = 0.2 * e[column_index] / np.mean(N2[select])
            diss['Krho'][index, column_index] = Krho
            diss['N2'][index, column_index] = np.mean(N2[select])
            k_phi, _ = nasmyth(e[column_index], nu, K)
            diss['Nasmyth_spec'][index, column_index, :] = k_phi
            flagood[column_index] = flagi

        # Save results
        diss['e'][index, :] = e
        diss['K_max'][index, :] = K_max
        diss['method'][index, :] = method
        diss['sh_clean'][index, :, :, :] = P_sh_clean
        diss['sh'][index, :, :, :] = P_sh
        diss['AA'][index, :, :, :] = AA
        diss['UA'][index, :, :, :] = UA.real
        diss['F'][index, :] = F
        diss['K'][index, :] = K
        diss['speed'][index, 0] = W
        diss['nu'][index, 0] = nu
        diss['T'][index, 0] = mean_T
        diss['P'][index, 0] = np.mean(P[select])
        diss['flagood'][index, :] = flagood

        index += 1
        select = select + int(diss_length - overlap)
        if select[-1] >= SH.shape[0]:
            break

    return diss
