import os
import json
import numpy as np
import pandas as pd
import segyio
import lasio
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
segy_path = os.path.abspath(os.path.join(script_dir, "..", "segy", "origional.segy"))
las_dir = os.path.abspath(os.path.join(script_dir, "..", "las"))
tie_summary_path = os.path.abspath(os.path.join(script_dir, "..", "well_seismic", "output", "tie_summary.csv"))
output_dir = os.path.abspath(os.path.join(script_dir, "output"))

# Create output folder
os.makedirs(output_dir, exist_ok=True)

FEET_TO_METERS = 0.3048
MAX_ENV_RATIO = 50.0  # Sane geological limit to prevent divide-by-zero blowout

def _pad_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 1D trace by one or more samples, padding with zeros at the edges."""
    arr = np.asarray(arr, dtype=float)
    if shift == 0:
        return arr.copy()
    if shift > 0:
        out = np.zeros_like(arr)
        out[:-shift] = arr[shift:]
        return out
    shift = -shift
    out = np.zeros_like(arr)
    out[shift:] = arr[:-shift]
    return out

def _rolling_stats(arr: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling max/mean/min/std around each sample."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    half = window // 2
    maxv = np.zeros(n, dtype=float)
    meanv = np.zeros(n, dtype=float)
    minv = np.zeros(n, dtype=float)
    stdv = np.zeros(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = arr[lo:hi]
        if len(window_vals) == 0:
            continue
        maxv[i] = float(np.max(window_vals))
        meanv[i] = float(np.mean(window_vals))
        minv[i] = float(np.min(window_vals))
        stdv[i] = float(np.std(window_vals))

    return maxv, meanv, minv, stdv

def compute_trace_attributes(trace, times_ms, dt_ms, global_ai_min=None, global_ai_max=None) -> dict[str, np.ndarray]:
    """Build a comprehensive 42-attribute feature dictionary from a seismic trace."""
    trace = np.asarray(trace, dtype=float)
    n_samples = len(trace)
    if n_samples == 0:
        return {}

    amp = np.nan_to_num(trace.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    analytic = hilbert(amp)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dt_s = dt_ms / 1000.0
    ifreq = np.gradient(phase, dt_s) / (2.0 * np.pi)
    ifreq = np.nan_to_num(ifreq, nan=0.0, posinf=0.0, neginf=0.0)

    rel_pos = np.linspace(0.0, 1.0, n_samples, dtype=float)
    env_deriv = np.nan_to_num(np.gradient(envelope), nan=0.0, posinf=0.0, neginf=0.0)

    # Relative impedance proxy
    impedance = np.cumsum(np.clip(amp, -1.0, 1.0))
    impedance = np.nan_to_num(impedance, nan=0.0, posinf=0.0, neginf=0.0)
    
    if global_ai_min is not None and global_ai_max is not None:
        # Global normalization scope
        impedance = (impedance - global_ai_min) / (global_ai_max - global_ai_min + 1e-8)
    else:
        # Fallback to local (per-well) normalization if not provided
        impedance = impedance - np.min(impedance)
        if np.max(impedance) > 0:
            impedance = impedance / np.max(impedance)

    # Dynamic epsilon to avoid near-zero division blowout in env_ratio
    max_abs_env = np.max(envelope)
    eps = 1e-6 * max_abs_env
    env_ratio = np.divide(envelope, np.abs(amp) + eps)
    env_ratio = np.clip(env_ratio, 0.0, MAX_ENV_RATIO)

    # Sweetness = envelope / sqrt(max(ifreq, 0.1))
    sweetness = envelope / np.sqrt(np.maximum(ifreq, 0.1))

    attrs = {
        "acoustic_impedance": impedance,
        "amp_center": amp,
        "amp_shift_+1": _pad_shift(amp, 1),
        "amp_shift_+2": _pad_shift(amp, 2),
        "amp_shift_+3": _pad_shift(amp, 3),
        "amp_shift_+4": _pad_shift(amp, 4),
        "amp_shift_+5": _pad_shift(amp, 5),
        "amp_shift_-1": _pad_shift(amp, -1),
        "amp_shift_-2": _pad_shift(amp, -2),
        "amp_shift_-3": _pad_shift(amp, -3),
        "amp_shift_-4": _pad_shift(amp, -4),
        "amp_shift_-5": _pad_shift(amp, -5),
        "env_center": envelope,
        "env_deriv": env_deriv,
        "env_ratio": env_ratio,
        "env_shift_+1": _pad_shift(envelope, 1),
        "env_shift_+2": _pad_shift(envelope, 2),
        "env_shift_+3": _pad_shift(envelope, 3),
        "env_shift_+4": _pad_shift(envelope, 4),
        "env_shift_+5": _pad_shift(envelope, 5),
        "env_shift_-1": _pad_shift(envelope, -1),
        "env_shift_-2": _pad_shift(envelope, -2),
        "env_shift_-3": _pad_shift(envelope, -3),
        "env_shift_-4": _pad_shift(envelope, -4),
        "env_shift_-5": _pad_shift(envelope, -5),
        "ifreq_center": ifreq,
        "ifreq_shift_+1": _pad_shift(ifreq, 1),
        "ifreq_shift_+2": _pad_shift(ifreq, 2),
        "ifreq_shift_+3": _pad_shift(ifreq, 3),
        "ifreq_shift_+4": _pad_shift(ifreq, 4),
        "ifreq_shift_+5": _pad_shift(ifreq, 5),
        "ifreq_shift_-1": _pad_shift(ifreq, -1),
        "ifreq_shift_-2": _pad_shift(ifreq, -2),
        "ifreq_shift_-3": _pad_shift(ifreq, -3),
        "ifreq_shift_-4": _pad_shift(ifreq, -4),
        "ifreq_shift_-5": _pad_shift(ifreq, -5),
        "sweetness": sweetness,
        "sweetness_shift_+1": _pad_shift(sweetness, 1),
        "sweetness_shift_+2": _pad_shift(sweetness, 2),
        "sweetness_shift_+3": _pad_shift(sweetness, 3),
        "sweetness_shift_+4": _pad_shift(sweetness, 4),
        "sweetness_shift_+5": _pad_shift(sweetness, 5),
        "sweetness_shift_-1": _pad_shift(sweetness, -1),
        "sweetness_shift_-2": _pad_shift(sweetness, -2),
        "sweetness_shift_-3": _pad_shift(sweetness, -3),
        "sweetness_shift_-4": _pad_shift(sweetness, -4),
        "sweetness_shift_-5": _pad_shift(sweetness, -5),
        "rel_pos": rel_pos,
    }

    win_max, win_mean, win_min, win_std = _rolling_stats(amp, window=5)
    attrs.update({
        "win_max": win_max,
        "win_mean": win_mean,
        "win_min": win_min,
        "win_std": win_std,
    })

    # Prefix all keys with attr_
    prefixed_attrs = {f"attr_{k}": v for k, v in attrs.items()}
    return prefixed_attrs

print("Starting training dataset generation with Block-Averaging...")

# Load well tie results
if not os.path.exists(tie_summary_path):
    raise FileNotFoundError(f"Tie summary not found at {tie_summary_path}. Run well-seismic tie first!")

df_ties = pd.read_csv(tie_summary_path)

with segyio.open(segy_path, "r", ignore_geometry=True) as f_segy:
    seis_times = f_segy.samples.astype(float)
    dt_ms = float(seis_times[1] - seis_times[0])
    
    all_data_rows = []
    
    # First pass: calculate raw unnormalized impedance across training wells to get global reference min/max
    print("\n[VERIFICATION] Verifying acoustic_impedance normalization scope...")
    print("Old scope: per-well (local normalization per trace)")
    print("New scope: global normalization (min/max calculated across training wells only, excluding blind well Z-04)")
    
    inlines = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
    crosslines = f_segy.attributes(segyio.TraceField.TraceNumber)[:]
    
    train_impedance_samples = []
    
    for _, row in df_ties.iterrows():
        well_name = row["well"]
        corr = row["correlation"]
        mapped_inline = int(row["inline"])
        mapped_crossline = int(row["crossline"])
        
        trace_idx = np.where((inlines == mapped_inline) & (crosslines == mapped_crossline))[0]
        if len(trace_idx) == 0:
            continue
        trace_idx = int(trace_idx[0])
        
        raw_trace = f_segy.trace[trace_idx].copy()
        polarity = -1.0 if corr < 0 else 1.0
        trace = raw_trace * polarity
        
        amp = np.nan_to_num(trace.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        impedance_raw = np.cumsum(np.clip(amp, -1.0, 1.0))
        impedance_raw = np.nan_to_num(impedance_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        if well_name != "Z-04":
            train_impedance_samples.extend(impedance_raw)
            
    global_ai_min = np.min(train_impedance_samples)
    global_ai_max = np.max(train_impedance_samples)
    print(f"[GLOBAL SCALE] Acoustic Impedance training min: {global_ai_min:.4f}, max: {global_ai_max:.4f} (excluding Z-04)")
    
    for _, row in df_ties.iterrows():
        well_name = row["well"]
        shift_ms = row["shift_ms"]
        corr = row["correlation"]
        mapped_inline = int(row["inline"])
        mapped_crossline = int(row["crossline"])
        
        # Find trace index matching inline/crossline
        trace_idx = np.where((inlines == mapped_inline) & (crosslines == mapped_crossline))[0]
        if len(trace_idx) == 0:
            continue
        trace_idx = int(trace_idx[0])
        
        # Get raw seismic trace and apply polarity correction
        raw_trace = f_segy.trace[trace_idx].copy()
        polarity = -1.0 if corr < 0 else 1.0
        trace = raw_trace * polarity
        
        # Generate 42 attributes using global normalization limits
        attrs = compute_trace_attributes(trace, seis_times, dt_ms, global_ai_min=global_ai_min, global_ai_max=global_ai_max)
        
        # Load well logs
        las_path = os.path.join(las_dir, f"{well_name}.las")
        if not os.path.exists(las_path):
            continue
            
        las = lasio.read(las_path)
        df_las = las.df()
        
        # Required curves
        target_curves = ["GR", "DT", "RHOB", "VSH", "PHIE", "SWE", "PHIT"]
        available_targets = [c for c in target_curves if c in df_las.columns]
        
        if "DPTM" not in df_las.columns:
            continue
            
        dptm_ms = df_las["DPTM"].values.astype(float)
        well_time_ms = dptm_ms + shift_ms
        
        # Assign each depth sample to the nearest seismic time bin index
        bin_indices = np.clip(
            ((well_time_ms - seis_times[0]) / dt_ms).round().astype(int),
            0, len(seis_times) - 1
        )
        
        # Perform block-averaging per seismic bin index
        well_blocked_rows = []
        for bin_idx in np.unique(bin_indices):
            mask = bin_indices == bin_idx
            t_ms = seis_times[bin_idx]
            
            row_dict = {
                "well_name": well_name,
                "seismic_time_ms": float(t_ms),
                "inline": mapped_inline,
                "crossline": mapped_crossline
            }
            
            # Average targets in this bin
            has_valid_target = False
            for curve_name in available_targets:
                vals = df_las[curve_name].values[mask].astype(float)
                mean_val = np.nanmean(vals)
                if not np.isnan(mean_val):
                    row_dict[f"target_{curve_name}"] = float(mean_val)
                    has_valid_target = True
                else:
                    row_dict[f"target_{curve_name}"] = float("nan")
                    
            if has_valid_target:
                # Add attributes at this time step
                for attr_name, attr_arr in attrs.items():
                    row_dict[attr_name] = float(attr_arr[bin_idx])
                well_blocked_rows.append(row_dict)
                
        df_well_blocked = pd.DataFrame(well_blocked_rows)
        all_data_rows.append(df_well_blocked)
        print(f"[SUCCESS] Prepared {len(df_well_blocked)} block-averaged samples for well {well_name}")

df_training_table = pd.concat(all_data_rows, ignore_index=True).dropna().reset_index(drop=True)
print(f"\nFinal training dataset contains {len(df_training_table)} block-averaged samples.")

# Save training table CSV
training_table_path = os.path.join(output_dir, "training_table.csv")
df_training_table.to_csv(training_table_path, index=False)
print(f"Saved training table to: {training_table_path}")

# --- Correlation Analysis for Top 15 Features ---
print("\nPerforming Feature Correlation Analysis...")
attr_cols = sorted([c for c in df_training_table.columns if c.startswith("attr_")])
target_cols = sorted([c for c in df_training_table.columns if c.startswith("target_")])

# Compute correlation matrix
corr_matrix = df_training_table[attr_cols + target_cols].corr()
sub_corr = corr_matrix.loc[attr_cols, target_cols]

# Save correlation coefficients to CSV
corr_csv_path = os.path.join(output_dir, "feature_correlations.csv")
sub_corr.to_csv(corr_csv_path)
print(f"Saved feature correlation coefficients to: {corr_csv_path}")

# Pick top 15 features based on average absolute correlation with targets
avg_abs_corr = sub_corr.abs().mean(axis=1).sort_values(ascending=False)
top_15_attrs = list(avg_abs_corr.index[:15])

# Filter sub_corr to top 15 for visualization
sub_corr_top15 = sub_corr.loc[top_15_attrs]

# Plot Heatmap
plt.figure(figsize=(10, 8))
im = plt.imshow(sub_corr_top15.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

# Add text annotations for correlation values
for i in range(sub_corr_top15.shape[0]):
    for j in range(sub_corr_top15.shape[1]):
        val = sub_corr_top15.values[i, j]
        plt.text(j, i, f"{val:+.3f}", ha="center", va="center", 
                 color="black" if abs(val) < 0.5 else "white", fontsize=10, weight="bold")

plt.colorbar(im)
plt.title("Correlation Heatmap: Top 15 Seismic Attributes vs. Well Log Targets", fontsize=12, weight="bold")
plt.yticks(range(len(top_15_attrs)), [c.replace("attr_", "") for c in top_15_attrs], fontsize=9)
plt.xticks(range(len(target_cols)), [c.replace("target_", "") for c in target_cols], fontsize=10)
plt.ylabel("Seismic Attribute (Features)", fontsize=11)
plt.xlabel("Well Log Property (Targets)", fontsize=11)
plt.tight_layout()

heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Saved top 15 correlation heatmap plot to: {heatmap_path}")

print("\nTop Correlations between Seismic Features and Well Logs:")
for log in target_cols:
    corrs = sub_corr[log].abs().sort_values(ascending=False)
    top_feature = corrs.index[0]
    top_val = sub_corr.loc[top_feature, log]
    print(f"  Target '{log.replace('target_', '')}': Best feature is '{top_feature.replace('attr_', '')}' with R = {top_val:+.3f}")

print("\nAll tasks completed successfully!")
