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
      'k_obs', 'P_shear_obs', 'P_nasmyth', 'k_min' 'k_max', 'best_epsilon',
      'window_index', 'probe_index', 'nu'
    """

    plot_spectra_interactive.saved_data = None

    # Initialize plot index
    plot_idx = {'current': 0}

    # Total number of plots
    num_plots = len(spectra_data)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Adjust to make room for buttons

    # We'll store the integration range and whether we are selecting start/end
    integration_clicks = {
        'k_start': None,
        'k_end': None,
        'selecting_start': False,
        'selecting_end': False
    }

    instruction_text = None  # Will hold any active instruction text

    # ----------------------------------------------------------------------
    # 1. Plot & Update
    # ----------------------------------------------------------------------
    def update_plot(index):
        ax.clear()
        data = spectra_data[index]

        k_obs = data['k_obs']
        P_shear_obs = data['P_shear_obs']
        P_nasmyth = data['P_nasmyth']
        k_min = data['k_min']
        k_max = data['k_max']
        best_epsilon = data['best_epsilon']
        window_index = data['window_index']
        probe_index = data['probe_index']
        nu = data['nu']

        # Plot observed shear spectrum
        ax.loglog(k_obs, P_shear_obs, linewidth=0.7,
                  color='r', label='Observed Shear Spectrum')

        # Plot Nasmyth spectrum for estimated epsilon
        ax.loglog(k_obs, P_nasmyth, 'k', linewidth=2,
                  label=f'Nasmyth (ε={best_epsilon:.2e} W/kg)')

        # Plot reference Nasmyth spectra
        p00_list, k00 = generate_reference_nasmyth_spectra(k_obs)
        for p00 in p00_list:
            ax.loglog(k00, p00, 'b--', linewidth=1)

        # Plot integration range lines
        ax.axvline(k_min, color='r', linestyle='--',
                   label='Integration Range Start')
        ax.axvline(k_max, color='g', linestyle='--',
                   label='Integration Range End')

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
            ax.text(0.5, 0.95, instruction_text.get_text(),
                    transform=ax.transAxes,
                    fontsize=12, color='blue',
                    ha='center', va='top')

        fig.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # 2. Recompute Epsilon
    # ----------------------------------------------------------------------
    def recompute_epsilon(index):
        """Given the new k_start/k_end, recalc epsilon, update the plot."""
        data = spectra_data[index]
        k_obs = data['k_obs']
        P_shear_obs = data['P_shear_obs']
        nu = data['nu']

        k_start = integration_clicks['k_start']
        k_end = integration_clicks['k_end']

        if k_start is None and k_end is None:
            return  # If both are unset, do nothing
        # Selecting k_end
        elif k_start is None and k_end is not None:
            k_start = data['k_min']
        # Selecing k_start
        elif k_start is not None and k_end is None:
            k_end = data['k_max']
        # Ensure k_start < k_end
        if k_start > k_end:
            k_start, k_end = k_end, k_start

        # Update best_k_range
        data['k_min'] = k_start
        data['k_max'] = k_end

        # Recompute best_epsilon using the new integration range
        data['best_epsilon'] = compute_epsilon(k_obs, P_shear_obs, nu,
                                               k_start, k_end)

        # Recompute Nasmyth spectrum with new epsilon
        data['P_nasmyth'], _ = nasmyth(data['best_epsilon'], nu, k_obs)

        # Update the plot
        update_plot(index)

    # ----------------------------------------------------------------------
    # 3. Click Handler (Selecting Start/End in the Axes)
    # ----------------------------------------------------------------------
    def on_click(event):
        """If user is selecting start or end, capture xdata as new k_start or k_end."""
        # Must be in axes
        if event.inaxes != ax:
            return

        print("Click at:", event.xdata, event.ydata)
        print(f"Selecting Start: {integration_clicks['selecting_start']}")
        print(f"Selecting End: {integration_clicks['selecting_end']}")

        # Are we selecting start or end?
        if integration_clicks['selecting_start']:
            k_click = event.xdata
            if k_click is not None:
                integration_clicks['k_start'] = k_click
                integration_clicks['selecting_start'] = False
                recompute_epsilon(plot_idx['current'])
                clear_instruction_text()
                fig.canvas.draw_idle()

        elif integration_clicks['selecting_end']:
            k_click = event.xdata
            if k_click is not None:
                integration_clicks['k_end'] = k_click
                integration_clicks['selecting_end'] = False
                recompute_epsilon(plot_idx['current'])
                clear_instruction_text()
                fig.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # 4. Buttons: Next/Prev
    # ----------------------------------------------------------------------
    def next_plot(event):
        if plot_idx['current'] < num_plots - 1:
            plot_idx['current'] += 1
            update_plot(plot_idx['current'])
            clear_instruction_text()

    def prev_plot(event):
        if plot_idx['current'] > 0:
            plot_idx['current'] -= 1
            update_plot(plot_idx['current'])
            clear_instruction_text()

    # ----------------------------------------------------------------------
    # 5. Buttons: Reselect Start/End, Save Results
    # ----------------------------------------------------------------------
    def reselect_start(event):
        integration_clicks['selecting_start'] = True
        integration_clicks['selecting_end'] = False
        set_instruction_text("Click to choose new integration START")

    def reselect_end(event):
        integration_clicks['selecting_start'] = False
        integration_clicks['selecting_end'] = True
        set_instruction_text("Click to choose new integration END")

    def on_save_results(event):
        # Here we can do whatever finalization or data extraction we need.
        # For example, store the final 'spectra_data' so code outside this function can read it.
        print("Saving final data and closing figure...")
        plot_spectra_interactive.saved_data = spectra_data

        # Then close the figure:
        plt.close(fig)

    # ----------------------------------------------------------------------
    # 6. Instruction Text Helpers
    # ----------------------------------------------------------------------
    def set_instruction_text(msg):
        nonlocal instruction_text
        clear_instruction_text()  # remove old text first
        instruction_text = ax.text(
            0.5, 0.95, msg, transform=ax.transAxes,
            fontsize=12, color='blue', ha='center', va='top'
        )
        fig.canvas.draw_idle()

    def clear_instruction_text():
        nonlocal instruction_text
        if instruction_text is not None:
            instruction_text.remove()
            instruction_text = None
            fig.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # 7. Place Buttons
    # ----------------------------------------------------------------------
    # Put the Previous button far left
    axprev = plt.axes([0.05, 0.05, 0.1, 0.075])
    # Put the Next button far right
    axnext = plt.axes([0.17, 0.05, 0.1, 0.075])

    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')

    bprev.on_clicked(prev_plot)
    bnext.on_clicked(next_plot)

    # Place the reselect buttons roughly in the center
    ax_reselect_start = plt.axes([0.38, 0.05, 0.18, 0.075])
    ax_reselect_end = plt.axes([0.58, 0.05, 0.18, 0.075])

    b_reselect_start = Button(ax_reselect_start, 'Reselect Start')
    b_reselect_end = Button(ax_reselect_end, 'Reselect End')

    b_reselect_start.on_clicked(reselect_start)
    b_reselect_end.on_clicked(reselect_end)

    # Connect the click event handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Place the "Save results" button
    ax_save = plt.axes([0.80, 0.05, 0.15, 0.075])
    b_save = Button(ax_save, "Save results")
    b_save.on_clicked(on_save_results)

    # Initialize the first plot
    update_plot(plot_idx['current'])
    plt.show()


def generate_reference_nasmyth_spectra(K):
    """
    Generate multiple reference Nasmyth spectra for
    a range of epsilon values, for visual reference.
    """
    epsilon_values = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    p00_list = []
    for e in epsilon_values:
        P_nasmyth, _ = nasmyth(e, 1e-6, K)
        p00_list.append(P_nasmyth)
    return p00_list, K


def compute_epsilon(K, P_sh, nu, k_min, k_max):
    """
    Compute epsilon from the integral of the shear spectrum
    over the specified k-range.
    """
    print(f"K: {K}")
    print(f"P_sh: {P_sh}")
    print(f"nu: {nu}")
    print(f"k_min: {k_min}")
    print(f"k_max: {k_max}")

    idx_integration = np.where((K >= k_min) & (K <= k_max))[0]
    print(f"idx_integration: {idx_integration}")

    epsilon = 7.5 * nu * np.trapz(P_sh[idx_integration], K[idx_integration])
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
    axs[1, 1].plot(history_dict['flagood_output_accuracy'], label='Train Acc')
    axs[1, 1].plot(history_dict['val_flagood_output_accuracy'],
                   label='Val Acc')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].set_title('Flagood Output Accuracy')
    axs[1, 1].legend()

    # Hide the last subplot (if not used)
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()
