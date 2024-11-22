import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from nasmyth import nasmyth


def plot_k_max(K_max):
    plt.hist(K_max, bins=20)
    plt.title("Distribution of K_max in Training Data")
    plt.xlabel('K_max')
    plt.ylabel('Frequency')
    plt.show()


def plot_spectra_interactive(spectra_data):
    """
    Plot multiple spectra in a single interactive window with navigation buttons.

    Parameters:
    - spectra_data (list of dict): A list where each element is a dictionary containing
      the data for one plot. Each dictionary should have the keys:
      'k_obs', 'P_shear_obs', 'P_nasmyth', 'best_k_range', 'best_epsilon', 'window_index', 'probe_index', 'nu'
    """
    # Initialize plot index
    plot_idx = {'current': 0}

    # Total number of plots
    num_plots = len(spectra_data)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)  # Adjust to make room for buttons

    # Function to update plot
    def update_plot(index):
        ax.clear()
        data = spectra_data[index]
        k_obs = data['k_obs']
        P_shear_obs = data['P_shear_obs']
        P_nasmyth = data['P_nasmyth']
        best_k_range = data['best_k_range']
        best_epsilon = data['best_epsilon']
        window_index = data['window_index']
        probe_index = data['probe_index']

        # Plot observed shear spectrum
        ax.loglog(k_obs, P_shear_obs, linewidth=0.7,
                  color='r', label='Observed Shear Spectrum')
        # Plot Nasmyth spectrum for estimated epsilon
        ax.loglog(k_obs, P_nasmyth, 'k', linewidth=2,
                  label=f'Nasmyth Spectrum (ε={best_epsilon:.2e} W/kg)')

        # Plot reference Nasmyth spectra
        p00_list, k00 = generate_reference_nasmyth_spectra(k_obs)
        for p00 in p00_list:
            ax.loglog(k00, p00, 'b--', linewidth=1)

        # Plot integration range
        ax.axvline(best_k_range[0], color='r',
                   linestyle='--', label='Integration Range Start')
        ax.axvline(best_k_range[1], color='g',
                   linestyle='--', label='Integration Range End')

        ax.set_xlabel('Wavenumber (cpm)')
        ax.set_ylabel('Shear Spectrum [(s$^{-1}$)$^2$/cpm]')
        ax.set_xlim([1, 1000])
        ax.set_ylim([1e-10, 1])
        ax.set_title(
            f'Shear Spectrum Fit (Window {window_index}, Probe {probe_index})')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.canvas.draw_idle()

    # Define button callbacks
    def next_plot(event):
        if plot_idx['current'] < num_plots - 1:
            plot_idx['current'] += 1
            update_plot(plot_idx['current'])

    def prev_plot(event):
        if plot_idx['current'] > 0:
            plot_idx['current'] -= 1
            update_plot(plot_idx['current'])

    # Add buttons
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')
    bnext.on_clicked(next_plot)
    bprev.on_clicked(prev_plot)

    # Initial plot
    update_plot(plot_idx['current'])
    plt.show()


def generate_reference_nasmyth_spectra(K):
    epsilon_values = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    p00_list = []
    for e in epsilon_values:
        P_nasmyth, _ = nasmyth(e, 1e-6, K)
        p00_list.append(P_nasmyth)
    return p00_list, K


def plot_training_history(history):
    import matplotlib.pyplot as plt

    # Extract the history data
    history_dict = history.history

    # Create subplots for losses and metrics
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # Plot overall loss
    axs[0, 0].plot(history_dict['loss'], label='Train Loss')
    axs[0, 0].plot(history_dict['val_loss'], label='Val Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Overall Loss')
    axs[0, 0].legend()

    # Plot loss for integration_output
    axs[0, 1].plot(history_dict['integration_output_loss'], label='Train Loss')
    axs[0, 1].plot(history_dict['val_integration_output_loss'],
                   label='Val Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Integration Output Loss')
    axs[0, 1].legend()

    # Plot loss for flagood_output
    axs[0, 2].plot(history_dict['flagood_output_loss'], label='Train Loss')
    axs[0, 2].plot(history_dict['val_flagood_output_loss'], label='Val Loss')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Loss')
    axs[0, 2].set_title('Flagood Output Loss')
    axs[0, 2].legend()

    # Plot metrics for integration_output (MAE)
    axs[1, 0].plot(history_dict['integration_output_mae'], label='Train MAE')
    axs[1, 0].plot(history_dict['val_integration_output_mae'], label='Val MAE')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Mean Absolute Error')
    axs[1, 0].set_title('Integration Output MAE')
    axs[1, 0].legend()

    # Plot accuracy for flagood_output
    axs[1, 1].plot(history_dict['flagood_output_accuracy'],
                   label='Train Accuracy')
    axs[1, 1].plot(history_dict['val_flagood_output_accuracy'],
                   label='Val Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].set_title('Flagood Output Accuracy')
    axs[1, 1].legend()

    # Hide the last subplot (if not used)
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()
