import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from columns import COLUMN_CONFIGS

matplotlib.use('TkAgg')

# ------------------------------
# Functions
# ------------------------------
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def multi_gaussian(x, *params):
    """Sum of multiple Gaussians for fallback fitting."""
    n = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n):
        a, mu, sigma = params[3*i:3*(i+1)]
        y += gaussian(x, a, mu, sigma)
    return y


def interactive_peak_fitting(x, y, void_volume=None, column_end=None):
    # Use the 21-point smoothing you liked for the visual/residual
    y_smooth = savgol_filter(y, 21, 3) if len(y) > 21 else y
    fits = []

    fig, ax = plt.subplots(figsize=(12, 6))

    def redraw():
        ax.clear()
        ax.plot(x, y_smooth, color='black', label='Chromatogram', alpha=0.8)

        total_fit = np.zeros_like(x)
        for i, f in enumerate(fits):
            peak_curve = gaussian(x, *f)
            total_fit += peak_curve
            ax.plot(x, peak_curve, '--', alpha=0.7, label=f'Peak {i + 1} @ {f[1]:.2f}')

        if len(fits) > 0:
            ax.plot(x, total_fit, 'r-', linewidth=1.5, label='Total Model', alpha=0.5)

        ax.set_title("1. Click Peaks (Largest to Smallest) | 2. Close Window for Global Refinement")
        ax.legend(fontsize='x-small', ncol=2, loc='upper right')
        ax.set_xlabel("Volume (mL)")
        ax.set_ylabel("Absorbance (mAU)")
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax or event.button != 1: return
        x_click = event.xdata

        # 1. Calculate Residual based on existing fits
        current_model = np.zeros_like(x)
        for f in fits:
            current_model += gaussian(x, *f)
        y_residual = y_smooth - current_model

        # 2. Fit to the local residual
        window = 0.5
        mask = (x >= x_click - window) & (x <= x_click + window)
        if len(x[mask]) < 5: return

        try:
            # Initial guess from residual
            a0 = max(y_residual[mask])
            mu0 = x[mask][np.argmax(y_residual[mask])]

            popt, _ = curve_fit(
                gaussian, x[mask], y_residual[mask],
                p0=[a0, mu0, 0.15],
                bounds=([0, x_click - 0.3, 0.05], [a0 * 2, x_click + 0.3, 0.8]),
                maxfev=5000
            )
            fits.append(popt)
            redraw()
        except Exception as e:
            print(f"❌ Residual fit failed: {e}")

    fig.canvas.mpl_connect('button_press_event', on_click)
    redraw()
    plt.show()  # Manual phase ends when window is closed
    plt.close(fig)  # <-- ADD THIS TO DESTROY THE GHOST

    # --- FINAL GLOBAL OPTIMIZATION ---
    # --- RESTRAINED GLOBAL OPTIMIZATION ---
    if len(fits) > 1:
        print("🧬 Refining peaks (Restricted Mode)...")
        p0 = [item for sublist in fits for item in sublist]

        low_b, high_b = [], []
        for f in fits:
            # ONLY allow the center (mu) to move 0.05 mL (one or two data points)
            # This keeps the peak tip exactly where you clicked it.
            low_b += [f[0] * 0.8, f[1] - 0.05, f[2] * 0.7]
            high_b += [f[0] * 1.2, f[1] + 0.05, f[2] * 1.3]

        try:
            popt_global, _ = curve_fit(
                multi_gaussian, x, y_smooth,
                p0=p0,
                bounds=(low_b, high_b),
                maxfev=5000
            )
            return [list(popt_global[i:i + 3]) for i in range(0, len(popt_global), 3)]
        except:
            return fits  # Fallback to your manual fits if it struggles

    return fits


# ------------------------------
# IO
# ------------------------------
def read_chromatogram(file_path):
    """
    Reads SEC data by automatically detecting the header row
    (looks for a column containing 'mAU').
    """

    # 1. Detect Encoding
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'cp1252', 'latin1']
    used_enc = 'latin1'

    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.readline()
            used_enc = enc
            break
        except Exception:
            continue

    # 2. Find header row dynamically
    header_row = None
    with open(file_path, 'r', encoding=used_enc) as f:
        for i, line in enumerate(f):
            if 'mAU' in line:  # key condition
                header_row = i
                break

    if header_row is None:
        raise ValueError("Could not find header row containing 'mAU'")

    # 3. Read with detected header
    df = pd.read_csv(
        file_path,
        sep=None,
        engine='python',
        encoding=used_enc,
        skiprows=header_row,   # skip everything before header
        on_bad_lines='skip'
    )

    # Auto-detect columns
    ml_col = next((c for c in df.columns if c.lower().startswith('ml')), None)
    mau_col = next((c for c in df.columns if 'mau' in c.lower()), None)

    if ml_col is None or mau_col is None:
        raise ValueError(f"Could not find ml/mAU columns. Found: {df.columns}")

    df[ml_col] = pd.to_numeric(df[ml_col], errors='coerce')
    df[mau_col] = pd.to_numeric(df[mau_col], errors='coerce')

    # Standardise names
    df = df.rename(columns={ml_col: 'ml', mau_col: 'mAU'})

    return df


def parse_akta_injection_volumes(file_path):
    """
    Robustly extracts injection volumes from AKTA exports.
    Works across different column layouts.
    """
    # 1. Detect Encoding
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'cp1252', 'latin1']
    used_enc = 'latin1'

    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.readline()
            used_enc = enc
            break
        except Exception:
            continue

    header_row = None
    with open(file_path, 'r', encoding=used_enc) as f:
        for i, line in enumerate(f):
            if 'Injection' in line:  # key condition
                header_row = i
                break

    if header_row is None:
        raise ValueError("Could not find header row containing 'injection'")

    # 3. Read with detected header
    df = pd.read_csv(
        file_path,
        sep='\t',
        engine='python',
        encoding=used_enc,
        skiprows=header_row,   # skip everything before header
        on_bad_lines='skip'
    )

    # Find injection-related columns
    injection_cols = [c for c in df.columns if 'inject' in c.lower()]

    if not injection_cols:
        return [0.0]

    col = injection_cols[0]
    col_idx = df.columns.get_loc(col)

    # --- Try parsing THIS column first ---
    vals = pd.to_numeric(df[col], errors='coerce').dropna()

    if len(vals) > 0:
        return vals.values

    # --- Otherwise fallback to previous column ---
    if col_idx > 0:
        vals_prev = pd.to_numeric(df.iloc[:, col_idx - 1], errors='coerce').dropna()
        if len(vals_prev) > 0:
            return vals_prev.values

    return [0.0]


def parse_akta_fractions(file_path):
    """
    Robustly extracts fraction names and their starting volumes from AKTA exports.
    Works for both 'Fraction' and '(Fractions)' column naming conventions.
    """
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'cp1252', 'latin1']
    used_enc = 'latin1'

    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.readline()
            used_enc = enc
            break
        except Exception:
            continue

    header_row = None
    with open(file_path, 'r', encoding=used_enc) as f:
        for i, line in enumerate(f):
            if 'mAU' in line and 'ml' in line:
                header_row = i
                break

    if header_row is None:
        return []

    # Read the data matrix
    df = pd.read_csv(
        file_path, sep='\t', engine='python', encoding=used_enc,
        skiprows=header_row, on_bad_lines='skip'
    )

    # Sometimes AKTA falls back to comma separated
    frac_col = next((c for c in df.columns if 'fraction' in c.lower()), None)
    if not frac_col:
        df = pd.read_csv(
            file_path, sep=None, engine='python', encoding=used_enc,
            skiprows=header_row, on_bad_lines='skip'
        )
        frac_col = next((c for c in df.columns if 'fraction' in c.lower()), None)

    if not frac_col:
        return []

    col_idx = df.columns.get_loc(frac_col)
    if col_idx == 0:
        return []

    # The column immediately to the left is always the relative volume (mL)
    vol_col = df.columns[col_idx - 1]

    df_frac = df[[vol_col, frac_col]].dropna()
    fractions = []

    for _, row in df_frac.iterrows():
        try:
            v = float(row[vol_col])
            name = str(row[frac_col]).strip(' "')  # Clean up surrounding quotes
            if name.lower() not in ['nan', 'none', '']:
                fractions.append((v, name))
        except ValueError:
            pass

    # Ensure fractions are sorted by volume
    fractions.sort(key=lambda x: x[0])
    return fractions

# ------------------------------
# Calibration
# ------------------------------

def fit_calibration_from_points(calib_points):

    Ve = np.array([p[0] for p in calib_points], dtype=float)
    Mw = np.array([p[1] for p in calib_points], dtype=float)

    a, b = np.polyfit(Ve, np.log10(Mw), 1)

    def compute(Ve_query):
        return 10 ** (a * np.array(Ve_query) + b)

    return a, b, compute


# ------------------------------
# Analysis
# ------------------------------

def analyze_sec(
    csv_path, injection_volume, analysis_window,
    calib_points=None, calib_chrom_csv=None, peak_prominence=0.1,
    baseline_fraction=(0.1, 0.3), void_volume=None, mu_cutoff=None,
    pre_void_fraction=0.10, expected_mw=None,
    fractions=None
):
    df = read_chromatogram(csv_path)

    x_raw = df["ml"].values
    x_raw = x_raw - injection_volume
    y_raw = df["mAU"].values

    plot_start = 0.0
    if void_volume is not None:
        plot_start = void_volume * (1 - pre_void_fraction)
        print("This is plot start: "+str(plot_start))

    mask = (x_raw >= plot_start) & (x_raw <= analysis_window)
    x = x_raw[mask]
    y = y_raw[mask]

    # Baseline correction
    start_frac, end_frac = baseline_fraction
    baseline_mask = (x >= analysis_window * start_frac) & (x <= analysis_window * end_frac)
    baseline_value = np.mean(y[baseline_mask]) if baseline_mask.sum() > 0 else 0.0
    y_corrected = y - baseline_value

    # --- FULL DATA (for export) ---
    mask_full = (x_raw >= 0) & (x_raw <= analysis_window)
    x_full = x_raw[mask_full]
    y_full = y_raw[mask_full]

    # Apply SAME baseline correction
    y_full_corrected = y_full - baseline_value

    # --- SAVE FULL DATA TO CSV ---
    base_name = os.path.splitext(csv_path)[0]
    export_name = f"{base_name}_processed_plot.csv"

    pd.DataFrame({
        "mL": x_full,
        "mAU_corrected": y_full_corrected
    }).to_csv(export_name, index=False)

    print(f"✅ Exported FULL plot data to: {export_name}")

    if calib_chrom_csv is not None:
        try:
            calib_df = read_chromatogram(calib_chrom_csv)

            calib_x = calib_df['ml'].values
            calib_y = calib_df['mAU'].values

            # apply same plotting window as experimental data
            plot_start = 0.0
            if void_volume is not None:
                plot_start = max(void_volume * (1 - pre_void_fraction), 0.0)

            calib_mask = (calib_x >= plot_start) & (calib_x <= analysis_window)

            calib_x = calib_x[calib_mask]
            calib_y = calib_y[calib_mask]

            scale_factor = (
                np.nanmax(y_corrected) / np.nanmax(calib_y)
                if np.nanmax(calib_y) != 0 else 1.0
            )

            calib_plot_x = calib_x
            calib_plot_y_scaled = calib_y * scale_factor

        except Exception as e:
            print(f"Could not read calibration chromatogram '{calib_chrom_csv}': {e}")

    interactive_fits = interactive_peak_fitting(
        x,
        y_corrected,
        void_volume=void_volume,
        column_end=analysis_window
    )
    compute_mw = None
    if calib_points is not None:
        _, _, compute_mw = fit_calibration_from_points(calib_points)

    results = []

    # 1. Calculate combined fit and individual areas first
    total_fit_y = np.zeros_like(x)
    areas = []
    for fit in interactive_fits:
        total_fit_y += gaussian(x, *fit)

        # Area = a * sigma * sqrt(2*pi)
        a_fit, mu, sigma = fit
        peak_area = a_fit * sigma * np.sqrt(2 * np.pi)
        areas.append(peak_area)

    total_area = sum(areas) if len(areas) > 0 else 1.0

    # Calculate Normalization Factor for bottom plots
    max_y = np.max(y_corrected) if np.max(y_corrected) != 0 else 1.0
    y_norm = y_corrected / max_y
    total_fit_y_norm = total_fit_y / max_y

    # Set up the figure conditionally (add 4th plot if fractions exist)
    if fractions is not None and len(fractions) > 0:
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.5])
        ax4 = fig.add_subplot(gs[2, :])
    else:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
        ax4 = None

    # --- AXIS 1 (Top Full Width): Raw Data + Peak Tops ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y_corrected, color='black', label='Experimental Data')
    ax1.set_title("Experimental Data with Identified Peaks")
    ax1.set_ylabel("Absorbance (mAU)")
    ax1.set_ylim(bottom=min(y_corrected) - (0.05 * max_y), top=max_y * 1.25)

    # --- AXIS 2 (Bottom Left): Calibration Overlay (Normalised) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, y_norm, color='black', label='Experimental Data', alpha=0.5)
    ax2.set_title("Calibration Overlay")
    ax2.set_ylabel("Normalised Absorbance")
    ax2.set_ylim(-0.05, 1.1)

    # --- AXIS 3 (Bottom Right): Gaussian Fits (Normalised) ---
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
    ax3.plot(x, y_norm, color='black', label='Experimental Data', alpha=0.5)
    ax3.set_title("Gaussian Deconvolution Fit")

    # Populate Ax1 (Dots/Mass) and Ax3 (Dashed curves)
    for i, fit in enumerate(interactive_fits):
        a_fit, mu, sigma = fit

        peak_curve = gaussian(x, *fit)
        ax3.plot(x, peak_curve / max_y, '--', alpha=0.7)

        y_actual = np.interp(mu, x, y_corrected)
        y_actual_norm = y_actual / max_y

        peak_area = areas[i]
        rel_abund = (peak_area / total_area) * 100
        stoichiometry = np.nan

        if mu_cutoff is not None and mu < mu_cutoff:
            ax1.plot(mu, y_actual, 'o', color='grey')
            ax2.plot(mu, y_actual_norm, 'o', color='grey')
            mw_val = np.nan
        else:
            ax1.plot(mu, y_actual, 'o', color='red')
            ax2.plot(mu, y_actual_norm, 'o', color='red')

            mw_val = compute_mw([mu])[0] if compute_mw else np.nan

            if compute_mw:
                annotation_text = f'{mw_val:.1f} kDa\n({mu:.2f} mL)'
                if expected_mw is not None and not np.isnan(mw_val):
                    stoichiometry = mw_val / expected_mw
                    annotation_text += f'\n{rel_abund:.1f}%, n={stoichiometry:.1f}'
                else:
                    annotation_text += f'\n{rel_abund:.1f}%'

                ax1.annotate(
                    annotation_text, xy=(mu, y_actual), xytext=(0, 20),
                    textcoords='offset points', ha='center', va='bottom', fontsize=9,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7)
                )

        results.append((mu, mu, mw_val, peak_area, rel_abund, stoichiometry))

    # Add combined fit line to Ax3 (Normalised)
    if len(interactive_fits) > 0:
        ax3.plot(x, total_fit_y_norm, 'r-', linewidth=1.5, label='Combined Fit')

    # Populate Ax2 (Calibration)
    if calib_plot_x is not None:
        ax2.plot(calib_plot_x, calib_plot_y_scaled / max_y, color='blue', alpha=0.3, label='Calibrant Data')
    if calib_points is not None and calib_plot_x is not None:
        for ve, mw in calib_points:
            y_interp = np.interp(ve, calib_plot_x, calib_plot_y_scaled)
            ax2.plot(ve, y_interp / max_y, 'o', color='blue', alpha=0.5)
            ax2.text(ve, (y_interp / max_y) + 0.03, f'{mw:.1f} kDa', ha='center', fontsize=8)

        # --- AXIS 4 (Bottom Full Width): Fractions (Optional) ---
        plot_start = 0.0
        if void_volume is not None:
            plot_start = void_volume * (1 - pre_void_fraction)

        if ax4 is not None:
            # 1. Determine zoom window based on fitted peaks
            if len(interactive_fits) > 0:
                min_mu = min(f[1] for f in interactive_fits)
                max_mu = max(f[1] for f in interactive_fits)
                max_sigma = max(f[2] for f in interactive_fits)
                buffer = max_sigma * 4  # Give a nice visual buffer around the peaks
                zoom_start = max(plot_start, min_mu - buffer)
                zoom_end = min(analysis_window, max_mu + buffer)
            else:
                zoom_start = plot_start
                zoom_end = analysis_window

            # 2. Scale Y axis dynamically to the zoomed region
            mask_zoom = (x >= zoom_start) & (x <= zoom_end)
            if mask_zoom.any():
                zoom_max_y = np.max(y_corrected[mask_zoom])
                zoom_min_y = np.min(y_corrected[mask_zoom])
                y_buffer = (zoom_max_y - zoom_min_y) * 0.15
                ax4.set_ylim(bottom=zoom_min_y - y_buffer, top=zoom_max_y + y_buffer)
                text_y = zoom_max_y + (y_buffer * 0.7)
            else:
                ax4.set_ylim(bottom=min(y_corrected) - (0.05 * max_y), top=max_y * 1.25)
                text_y = max_y * 1.15

            # 3. Plot the data
            ax4.plot(x, y_corrected, color='black', label='Experimental Data', alpha=0.8)
            if len(interactive_fits) > 0:
                ax4.plot(x, total_fit_y, 'r-', linewidth=1.5, label='Combined Fit')
                for fit in interactive_fits:
                    ax4.plot(x, gaussian(x, *fit), '--', alpha=0.5)

            ax4.set_title("Fraction Collection (Zoomed to Peaks)")
            ax4.set_ylabel("Absorbance (mAU)")
            ax4.set_xlim(zoom_start, zoom_end)  # Apply the zoom!

            # 4. Map the fractions
            adj_fractions = [(v - injection_volume, name) for v, name in fractions]

            # Pre-calculate intervals so we know exactly where each fraction ends
            frac_intervals = []
            for i, (v, name) in enumerate(adj_fractions):
                v_next = adj_fractions[i + 1][0] if i + 1 < len(adj_fractions) else v + 2.0
                frac_intervals.append((v, v_next, name))

            # Filter for fractions that overlap our zoom window
            valid_fracs = [(v, v_next, n) for v, v_next, n in frac_intervals if v_next > zoom_start and v < zoom_end]

            for i, (v, v_next, name) in enumerate(valid_fracs):
                ax4.axvline(v, color='green', linestyle='-', alpha=0.3)
                ax4.axvspan(v, v_next, color='green', alpha=0.08 if i % 2 == 0 else 0.0)

                # Center the text in the *visible* portion of the fraction tube
                vis_start = max(v, zoom_start)
                vis_end = min(v_next, zoom_end)
                mid_v = (vis_start + vis_end) / 2

                # clip_on=True ensures text doesn't spill past the graph borders
                ax4.text(mid_v, text_y, name, rotation=90,
                         ha='center', va='top', fontsize=8, color='darkgreen', clip_on=True)

    # --- Formatting for all axes ---
    axes_list = [ax1, ax2, ax3]
    if ax4 is not None:
        axes_list.append(ax4)

    for ax in axes_list:
        # Add column markers
        if void_volume is not None:
            ax.axvline(void_volume, color='purple', linestyle='--', alpha=0.5, label='Void Vol')
        ax.axvline(analysis_window, color='gray', linestyle=':', alpha=0.5, label='Col End')

        # ONLY apply full-width xlim to the top three plots
        if ax != ax4:
            ax.set_xlim(plot_start, analysis_window)

        ax.set_xlabel("Elution Volume (mL)")

        # Generate clean legends without duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize='x-small')

    plt.tight_layout()
    plt.show()
    plt.close(fig)  # Prevents the ghost window

    # --- EXPORT TO CSV ---
    base_name = os.path.splitext(csv_path)[0]
    export_name = f"{base_name}_processed_plot.csv"

    export_data = {
        "Volume_mL": x,
        "mAU_Raw": y,
        "mAU_Baseline_Corrected": y_corrected
    }

    if len(interactive_fits) > 0:
        export_data["mAU_Combined_Fit"] = total_fit_y
        for i, fit in enumerate(interactive_fits):
            export_data[f"mAU_Peak_{i + 1}"] = gaussian(x, *fit)

    pd.DataFrame(export_data).to_csv(export_name, index=False)
    print(f"✅ Exported data AND model fits to: {export_name}")

    # Build and print the result dataframe
    result_df = pd.DataFrame(
        results,
        columns=[
            "Peak_Ve_rel (mL)",
            "Peak_Ve_abs (mL)",
            "Molecular_Weight (kDa)",
            "Peak_Area",
            "Relative_Abundance (%)",
            "Est_Stoichiometry (n)"
        ]
    )

    print("\n--- FINAL RESULTS ---")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notnull(x) else "NaN"))

    return result_df


# ------------------------------
# CLI
# ------------------------------

import argparse

if __name__ == "__main__":

    SCRIPT_VERSION = "2.0.0-multicolumn"

    print(f"\nSEC analysis script version: {SCRIPT_VERSION}\n")

    parser = argparse.ArgumentParser(
        description="Analyze an analytical SEC chromatogram."
    )
    parser.add_argument("csv_path")
    args = parser.parse_args()

    print("\nSelect column used:")
    for k, v in COLUMN_CONFIGS.items():
        print(f"[{k}] {v['name']}")

    while True:
        try:
            col_choice = int(input("Enter column index: "))
            if col_choice in COLUMN_CONFIGS:
                column_cfg = COLUMN_CONFIGS[col_choice]
                break
        except ValueError:
            pass

    print(f"\nUsing column: {column_cfg['name']}")

    # ---------------------------------
    # Injection volume selection
    # ---------------------------------

    injection_volumes = parse_akta_injection_volumes(args.csv_path)

    print("\nAvailable injection volumes:")
    for i, v in enumerate(injection_volumes):
        print(f"[{i}] {v:.3f} mL")

    while True:
        try:
            choice = int(input("Enter the index of the injection volume to use: "))
            if 0 <= choice < len(injection_volumes):
                injection_volume = injection_volumes[choice]
                break
        except ValueError:
            pass

    # ---------------------------------
    # Expected MW input
    # ---------------------------------
    print("\n(Optional) Estimate Stoichiometry:")
    mw_input = input("Enter expected monomer MW in kDa (or press Enter to skip): ")
    try:
        user_expected_mw = float(mw_input) if mw_input.strip() else None
    except ValueError:
        print("Invalid input. Skipping stoichiometry estimation.")
        user_expected_mw = None

    # ---------------------------------
    # Fraction plotting selection
    # ---------------------------------
    fractions_found = parse_akta_fractions(args.csv_path)
    user_fractions = None
    if len(fractions_found) > 0:
        print(f"\nFound {len(fractions_found)} fractions in the dataset.")
        plot_frac = input("Would you like to plot them on a fourth graph? (y/n) [y]: ").strip().lower()
        if plot_frac == '' or plot_frac == 'y':
            user_fractions = fractions_found

    # ---------------------------------
    # Run analysis
    # ---------------------------------
    print("\nRun parameters:")
    for k, v in column_cfg.items():
        if k != "calib_points":
            print(f"  {k}: {v}")

    if user_expected_mw is not None:
        print(f"  expected_mw: {user_expected_mw} kDa")

    analyze_sec(
        csv_path=args.csv_path,
        injection_volume=injection_volume,
        analysis_window=column_cfg["analysis_window"],
        calib_points=column_cfg["calib_points"],
        calib_chrom_csv=column_cfg["calib_csv"],
        peak_prominence=2,
        baseline_fraction=(0.1, 0.3),
        void_volume=column_cfg["void_volume"],
        mu_cutoff=column_cfg["mu_cutoff"],
        pre_void_fraction=column_cfg["pre_void_fraction"],
        expected_mw=user_expected_mw,
        fractions=user_fractions  # <-- Pass it in here
    )