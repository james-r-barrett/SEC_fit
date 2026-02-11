import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

matplotlib.use('TkAgg')  # keep your GUI backend


# ------------------------------
# Helper Functions
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


def interactive_peak_fitting(x, y, existing_fits=None, smooth=True):

    if smooth and len(y) > 11:
        y_smooth = savgol_filter(y, 11, 3)
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
    ax.legend()

    def on_click(event):

        if event.inaxes != ax:
            return

        x_click = event.xdata

        window = max((x[-1] - x[0]) / 40, 0.5)
        mask = (x >= x_click - window) & (x <= x_click + window)
        x_window, y_window = x[mask], y_smooth[mask]

        if len(x_window) < 3:
            print(f"Not enough points around {x_click:.2f} mL.")
            return

        a0 = max(y_window)
        mu0 = x_window[np.argmax(y_window)]
        sigma0 = max((x_window[-1] - x_window[0]) / 4.0, 0.01)

        try:
            popt, _ = curve_fit(
                gaussian,
                x_window,
                y_window,
                p0=[a0, mu0, sigma0],
                maxfev=4000
            )
            fits.append(popt)
            ax.plot(x, gaussian(x, *popt), '--', label=f'Peak @ {popt[1]:.2f} mL')
            print(f"✅ Single Gaussian fit at {popt[1]:.2f} mL")

        except Exception:

            peaks, _ = find_peaks(y_window, prominence=(a0*0.05))

            if len(peaks) >= 2:
                print(f"⚠️  Trying double Gaussian fit near {x_click:.2f} mL")

                params0 = []
                for px in x_window[peaks[:2]]:
                    params0 += [max(y_window), px, sigma0]

                try:
                    popt2, _ = curve_fit(
                        multi_gaussian,
                        x_window,
                        y_window,
                        p0=params0,
                        maxfev=8000
                    )
                    fits.append(popt2[:3])
                    fits.append(popt2[3:6])
                    ax.plot(x, multi_gaussian(x, *popt2), '--')
                    print(f"✅ Double Gaussian fit near {x_click:.2f} mL")

                except Exception as e:
                    print(f"❌ Could not fit multiple peaks near {x_click:.2f}: {e}")
            else:
                print(f"❌ Failed to fit around {x_click:.2f} mL")

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    print("Click on the chromatogram to add peaks. Close the window when done.")
    plt.show()

    return fits


# ------------------------------
# IO
# ------------------------------

def read_chromatogram(file_path):

    try:
        with open(file_path, 'r', encoding='utf-16') as f:
            lines = f.readlines()
            encoding = 'utf-16'
    except UnicodeError:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            encoding = 'utf-8'

    start_idx = None

    for i, line in enumerate(lines):

        parts = line.replace(',', '\t').strip().split('\t')

        if len(parts) < 2:
            continue

        try:
            float(parts[0])
            float(parts[1])
            start_idx = i
            break
        except ValueError:
            continue

    if start_idx is None:
        raise ValueError("Could not find where numeric data starts in the file.")

    df = pd.read_csv(
        file_path,
        sep=None,
        engine='python',
        encoding=encoding,
        skiprows=start_idx
    )

    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if df.shape[1] < 2:
        raise ValueError("File still doesn't have two numeric columns — check format.")

    df = df.iloc[:, :2]
    df.columns = ["ml", "mAU"]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    return df


def parse_akta_injection_volumes(file_path):

    try:
        df = pd.read_csv(file_path, encoding='utf-16', sep=None, engine='python', skiprows=2)
    except UnicodeError:
        df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python', skiprows=2)

    col_candidates = [c for c in df.columns if "Injection" in c]

    if not col_candidates:
        raise ValueError("No 'Injection' column found in the file.")

    col_idx = df.columns.get_loc(col_candidates[0])

    if col_idx == 0:
        raise ValueError("No column to the left of the 'Injection' column.")

    values = pd.to_numeric(df.iloc[:, col_idx - 1], errors='coerce').dropna().values

    return values


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
    csv_path,
    injection_volume,
    analysis_window,
    injection_size,
    calib_points=None,
    calib_chrom_csv=None,
    peak_prominence=0.1,
    baseline_fraction=(0.1, 0.3)
):

    df = read_chromatogram(csv_path)

    x = df["ml"].values - injection_volume + injection_size
    mask = (x >= 0) & (x <= analysis_window)

    x = x[mask]
    y = df["mAU"].values[mask]

    start_frac, end_frac = baseline_fraction

    baseline_mask = (x >= analysis_window * start_frac) & (x <= analysis_window * end_frac)

    baseline_value = np.mean(y[baseline_mask]) if baseline_mask.sum() > 0 else 0.0
    y_corrected = y - baseline_value

    calib_plot_x = calib_plot_y_scaled = None

    if calib_chrom_csv is not None:
        try:
            calib_df = read_chromatogram(calib_chrom_csv)

            calib_x = calib_df['ml'].values
            calib_y = calib_df['mAU'].values

            scale_factor = (
                np.nanmax(y_corrected) / np.nanmax(calib_y)
                if np.nanmax(calib_y) != 0 else 1.0
            )

            calib_plot_x = calib_x
            calib_plot_y_scaled = calib_y * scale_factor

        except Exception as e:
            print(f"Could not read calibration chromatogram '{calib_chrom_csv}': {e}")

    interactive_fits = interactive_peak_fitting(x, y_corrected)

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

        if mu < 9.0:
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

    plt.axvline(8.51, color='purple', linestyle='--', label='Void volume')
    plt.axvline(0, color='red', linestyle='--', label='Injection')
    plt.axvline(analysis_window, color='gray', linestyle=':', label='Column End')

    plt.xlabel("Elution Volume (mL) relative to injection")
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

    parser = argparse.ArgumentParser(
        description="Analyze an analytical SEC chromatogram."
    )
    parser.add_argument("csv_path")
    args = parser.parse_args()

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

    calib_points = [ ## from 50 mM Tris-HCl, 500 mM NaCl, pH 8.0
        (9.11, 669),
        (9.85, 550),
        (11.99, 150),
        (15, 44.3),
        (17.5, 13.7)
    ]

    analyze_sec(
        csv_path=args.csv_path,
        injection_volume=injection_volume,
        analysis_window=24.0,
        injection_size=0.1,
        calib_points=calib_points,
        calib_chrom_csv="calibration.csv",
        peak_prominence=2,
        baseline_fraction=(0.1, 0.3)
    )