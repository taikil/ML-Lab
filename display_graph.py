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
    plt.subplots_adjust(bottom=0.25)  # Adjust to make room for buttons
    integration_clicks = {'click_count': 0, 'k_start': None, 'k_end': None}
    selecting_range = False  # Flag to indicate if we're selecting a new range
    instruction_text = None  # To store the instruction text object

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
        nu = data['nu']

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

        # If instruction text exists, redraw it
        if instruction_text is not None:
            ax.text(0.5, 0.95, instruction_text.get_text(), transform=ax.transAxes,
                    fontsize=12, color='blue', ha='center', va='top')

        fig.canvas.draw_idle()

    # Function to recompute epsilon and update the plot
    def recompute_epsilon(index):
        data = spectra_data[index]
        k_obs = data['k_obs']
        P_shear_obs = data['P_shear_obs']
        nu = data['nu']
        k_start = integration_clicks['k_start']
        k_end = integration_clicks['k_end']

        # Update best_k_range
        data['best_k_range'] = [k_start, k_end]

        # Recompute best_epsilon using the new integration range
        # Implement your method to compute epsilon here
        # For example:
        data['best_epsilon'] = compute_epsilon(
            k_obs, P_shear_obs, nu, k_start, k_end)

        # Recompute Nasmyth spectrum with new epsilon
        data['P_nasmyth'], _ = nasmyth(data['best_epsilon'], nu, k_obs)

        # Update the plot
        update_plot(index)

    # Define click event handler
    def on_click(event):
        nonlocal selecting_range, instruction_text
        if not selecting_range:
            return  # Ignore clicks if not selecting range

        # Only consider clicks within the plot area
        if event.inaxes != ax:
            return

        # Get the x-coordinate of the click (wavenumber)
        k_click = event.xdata

        if k_click is None:
            return

        # Update click count and integration range
        if integration_clicks['click_count'] == 0:
            integration_clicks['k_start'] = k_click
            integration_clicks['click_count'] = 1
            print(f"Selected integration range start: {k_click:.2f} cpm")
        elif integration_clicks['click_count'] == 1:
            integration_clicks['k_end'] = k_click
            integration_clicks['click_count'] = 2
            print(f"Selected integration range end: {k_click:.2f} cpm")

            # Ensure k_start < k_end
            if integration_clicks['k_start'] > integration_clicks['k_end']:
                integration_clicks['k_start'], integration_clicks['k_end'] = integration_clicks['k_end'], integration_clicks['k_start']

            # Recompute epsilon and update plot
            recompute_epsilon(plot_idx['current'])

            # Reset click count for next adjustment
            integration_clicks['click_count'] = 0
            selecting_range = False  # Disable range selection

            # Remove instruction text
            if instruction_text is not None:
                instruction_text.remove()
                instruction_text = None

            fig.canvas.draw_idle()
        else:
            # Reset if more than two clicks
            integration_clicks['click_count'] = 0

    # Define button callbacks
    def next_plot(event):
        nonlocal selecting_range, instruction_text
        if plot_idx['current'] < num_plots - 1:
            plot_idx['current'] += 1
            update_plot(plot_idx['current'])
            # Reset selection
            selecting_range = False
            if instruction_text is not None:
                instruction_text.remove()
                instruction_text = None

    def prev_plot(event):
        nonlocal selecting_range, instruction_text
        if plot_idx['current'] > 0:
            plot_idx['current'] -= 1
            update_plot(plot_idx['current'])
            # Reset selection
            selecting_range = False
            if instruction_text is not None:
                instruction_text.remove()
                instruction_text = None

    def reselect_range(event):
        nonlocal selecting_range, instruction_text
        selecting_range = True
        integration_clicks['click_count'] = 0
        integration_clicks['k_start'] = None
        integration_clicks['k_end'] = None

        # Remove previous instruction text if it exists
        if instruction_text is not None:
            instruction_text.remove()

        # Display instruction text
        instruction = 'First click K_min, second click K_max'
        instruction_text = ax.text(0.5, 0.95, instruction, transform=ax.transAxes,
                                   fontsize=12, color='blue', ha='center', va='top')

        fig.canvas.draw_idle()

    # Add buttons
    axprev = plt.axes([0.5, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.61, 0.05, 0.1, 0.075])
    axreselect = plt.axes([0.72, 0.05, 0.15, 0.075])

    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')
    breselect = Button(axreselect, 'Reselect Range')

    bprev.on_clicked(prev_plot)
    bnext.on_clicked(next_plot)
    breselect.on_clicked(reselect_range)

    # Connect the click event handler
    fig.canvas.mpl_connect('button_press_event', on_click)

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


def compute_epsilon(K, P_sh, nu, k_max, k_min):
    idx_integration = np.where(
        (K >= k_min) & (K <= k_max))[0]
    epsilon = 7.5 * nu * \
        np.trapz(P_sh[idx_integration],
                 K[idx_integration])
    return epsilon


def plot_training_history(history):
    import matplotlib.pyplot as plt

    # Extract the history data
    history_dict = history.history

    # Create subplots for losses and metrics
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

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
