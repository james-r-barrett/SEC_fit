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

def interactive_peak_fitting(
    x,
    y,
    existing_fits=None,
    smooth=True,
    void_volume=None,
    column_end=None
):
    if smooth and len(y) > 21:
        y_smooth = savgol_filter(y, 21, 3)
    else:
        y_smooth = y

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_smooth, color='black', label='Chromatogram')

    fits = [] if existing_fits is None else list(existing_fits)

    for fit in fits:
        ax.plot(x, gaussian(x, *fit), '--', label=f'Peak @ {fit[1]:.2f} mL')

    ax.set_xlabel("Elution Volume (mL)")
    ax.set_ylabel("Absorbance (mAU)")
    ax.set_title("Click on missed peaks to fit Gaussian(s). Close window when done.")

    if void_volume is not None:
        ax.axvline(void_volume, color='purple', linestyle='--', label='Void volume')

    if column_end is not None:
        ax.axvline(column_end, color='gray', linestyle=':', label='Column End')

    ax.legend()

    def on_click(event):
        if event.inaxes != ax:
            return

        x_click = event.xdata
        # Narrow the window slightly for messy data to focus on the specific shoulder
        window = max((x[-1] - x[0]) / 50, 0.3)
        mask = (x >= x_click - window) & (x <= x_click + window)
        x_window, y_window = x[mask], y_smooth[mask]

        if len(x_window) < 5:
            return

        a0 = max(y_window)
        mu0 = x_window[np.argmax(y_window)]
        # Initial guess for sigma (peak width)
        sigma0 = 0.2

        try:
            # BOUNDS: [min_a, min_mu, min_sigma], [max_a, max_mu, max_sigma]
            # Sigma is restricted between 0.05mL (sharp) and 1.5mL (broad)
            lower_b = [0, x_click - 0.5, 0.05]
            upper_b = [a0 * 2, x_click + 0.5, 1.5]

            popt, _ = curve_fit(
                gaussian,
                x_window,
                y_window,
                p0=[a0, mu0, sigma0],
                bounds=(lower_b, upper_b),
                maxfev=5000
            )
            fits.append(popt)
            ax.plot(x, gaussian(x, *popt), '--', linewidth=1.5, label=f'Peak @ {popt[1]:.2f}')
            print(f"✅ Fit peak at {popt[1]:.2f} mL")

        except Exception as e:
            print(f"❌ Fit failed: {e}")

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    print("Click on the chromatogram to add peaks. Close the window when done.")
    plt.show()

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
    pre_void_fraction=0.10
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

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_corrected, color='black', label='Experimental chromatogram')

    if calib_plot_x is not None:
        plt.plot(
            calib_plot_x,
            calib_plot_y_scaled,
            color='blue',
            alpha=0.25,
            label='Calibrant chromatogram'
        )

    for fit in interactive_fits:

        a_fit, mu, sigma = fit
        plt.plot(x, gaussian(x, *fit), '--', label=f'Gaussian @ {mu:.2f} mL')

        y_peak = gaussian(mu, *fit)

        if mu_cutoff is not None and mu < mu_cutoff:
            plt.plot(mu, y_peak, 'o', color='grey')
            mw_val = np.nan
        else:
            plt.plot(mu, y_peak, 'o', color='red')
            mw_val = compute_mw([mu])[0] if compute_mw else np.nan

            if compute_mw:
                plt.text(
                    mu,
                    y_peak + 0.02 * np.max(y_corrected),
                    f'{mw_val:.1f}',
                    ha='center',
                    fontsize=9
                )

        results.append((mu, mu, mw_val))

    if calib_points is not None and calib_plot_x is not None:
        for ve, mw in calib_points:
            y_interp = np.interp(ve, calib_plot_x, calib_plot_y_scaled)
            plt.plot(ve, y_interp, 'o', color='blue', alpha=0.25)
            plt.text(
                ve,
                y_interp + 0.02 * np.max(y_corrected),
                f'{mw:.1f}',
                ha='center',
                fontsize=9
            )

    if void_volume is not None:
        plt.axvline(void_volume, color='purple', linestyle='--', label='Void volume')
    plt.axvline(analysis_window, color='gray', linestyle=':', label='Column End')

    # enforce identical x-axis start
    plot_start = 0.0
    if void_volume is not None:
        plot_start = void_volume * (1 - pre_void_fraction)

    plt.xlim(plot_start, analysis_window)

    plt.xlabel("Elution Volume (mL)")
    plt.ylabel("Absorbance (mAU, baseline-corrected)")
    plt.legend()
    plt.show()

    result_df = pd.DataFrame(
        results,
        columns=["Peak_Ve_rel (mL)", "Peak_Ve_abs (mL)", "Molecular_Weight (Da)"]
    )

    print(result_df.to_string(index=False))

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
    # Run analysis
    # ---------------------------------
    print("\nRun parameters:")
    for k, v in column_cfg.items():
        if k != "calib_points":
            print(f"  {k}: {v}")

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
        pre_void_fraction=column_cfg["pre_void_fraction"]
    )