import numpy as np


def nasmyth(*args):
    """
    Generate a Nasmyth universal shear spectrum.

    Usage:

    1) phi, k = nasmyth(e, nu=1e-6, N=1000)
       - Returns the dimensional Nasmyth spectrum for the dissipation rate `e` and kinematic viscosity `nu`.
       - The spectrum `phi` and wavenumber `k` are arrays of length `N`.

    2) phi, k = nasmyth(e, nu=1e-6, k_array)
       - Evaluates the Nasmyth spectrum at the wavenumbers provided in `k_array`.

    3) phi, k = nasmyth(0, N=1000)
       - Returns the non-dimensional Nasmyth spectrum `G2` of length `N`.
       - The wavenumber `k` is in terms of the non-dimensional form `k / k_s`.

    4) phi, k = nasmyth(0, k_array)
       - Evaluates the non-dimensional Nasmyth spectrum at the wavenumbers provided in `k_array`.

    Parameters:
        e (float or array-like): Dissipation rate(s) in W/kg. Use 0 for non-dimensional spectrum.
        nu (float): Kinematic viscosity in m^2/s. Default is 1e-6.
        N (int): Number of spectral points. Default is 1000.
        k_array (array-like): Wavenumber array in cpm (cycles per meter).

    Returns:
        phi (ndarray): Nasmyth spectrum values.
        k (ndarray): Corresponding wavenumber values in cpm.

    Notes:
        - If `e` is provided and non-zero, the function returns the dimensional spectrum.
        - If `e` is zero, the function returns the non-dimensional spectrum.
        - If `k_array` is provided, the spectrum is evaluated at those wavenumbers.
        - The function supports vectorized computations for multiple dissipation rates.

    References:
        - Oakey, N. S., 1982: J. Phys. Ocean., 12, 256-271.
        - McMillan, J.M., A.E. Hay, R.G. Lueck and F. Wolk, 2016: Rates of
          dissipation of turbulent kinetic energy in a high Reynolds Number tidal
          channel. J. Atmos. and Oceanic Technology, 33.
    """

    # Default values
    nu = 1e-6
    N = 1000
    k = None

    # Check the number of arguments
    if len(args) == 0:
        raise ValueError("At least one argument (e) is required.")

    e = np.atleast_1d(args[0])

    # Determine if the spectrum is dimensional or non-dimensional
    if np.any(e != 0):
        # Dimensional spectrum (Forms 1 and 2)
        scaled = True
        if len(args) >= 2:
            if np.isscalar(args[1]):
                nu = args[1]
                if len(args) == 3:
                    if np.isscalar(args[2]):
                        N = args[2]
                        k = None
                    else:
                        k = np.atleast_1d(args[2])
                        N = len(k)
                else:
                    k = None
            else:
                k = np.atleast_1d(args[1])
                N = len(k)
        else:
            k = None

        # Compute the Kolmogorov wavenumber(s)
        ks = (e / nu**3) ** 0.25  # Shape: (Ne,)
        ks = ks.reshape(1, -1)    # Shape: (1, Ne)
        Ne = ks.shape[1]

        if k is None:
            # Generate wavenumber array if not provided
            N = int(N)
            x = np.logspace(-4, 0, N).reshape(N, 1)  # Shape: (N, 1)
            x = np.tile(x, (1, Ne))                  # Shape: (N, Ne)
            k = x * ks                               # Shape: (N, Ne)
        else:
            k = k.reshape(N, 1)
            k = np.tile(k, (1, Ne))
            x = k / ks

        # Compute the Nasmyth spectrum
        G2 = 8.05 * x ** (1/3) / (1 + (20.6 * x) ** 3.715)
        phi = e ** (0.75) * nu ** (-0.25) * G2  # Shape: (N, Ne)

        # Flatten outputs if only one dissipation rate
        if Ne == 1:
            phi = phi.flatten()
            k = k.flatten()

        return phi, k

    else:
        # Non-dimensional spectrum (Forms 3 and 4)
        scaled = False
        if len(args) >= 2:
            if np.isscalar(args[1]):
                N = args[1]
                N = int(N)
                k = np.logspace(-4, 0, N)
            else:
                k = np.atleast_1d(args[1])
                N = len(k)
        else:
            k = np.logspace(-4, 0, N)

        G2 = 8.05 * k ** (1/3) / (1 + (20.6 * k) ** 3.715)
        phi = G2

        return phi, k
