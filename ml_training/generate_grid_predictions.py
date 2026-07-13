import os, sys, json, time
import numpy as np
import segyio
import joblib
from scipy.signal import hilbert

# ── paths ──────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
segy_path    = os.path.join(analysis_dir, "segy",          "origional.segy")
models_dir   = os.path.join(analysis_dir, "ml_outputs_v3", "models")
output_js    = os.path.join(analysis_dir, "frontend", "src", "grid_data.js")

# Reservoir window to average predictions over (ms TWT)
RESERVOIR_T_START = 2086.0
RESERVOIR_T_END   = 2154.0

MAX_ENV_RATIO = 50.0
FEET_TO_METERS = 0.3048

WELLS = {
    "Z-02": {"x": 1205859.09, "y": 9692966.31, "kb": 146.46},
    "Z-03": {"x": 1201178.25, "y": 9682452.00, "kb": 147.64},
    "Z-04": {"x": 1205820.18, "y": 9696292.65, "kb": 147.64},
    "Z-05": {"x": 1206404.17, "y": 9679510.83, "kb": 144.36},
    "Z-06": {"x": 1207337.37, "y": 9684145.64, "kb": 146.98},
    "Z-07": {"x": 1206364.34, "y": 9688320.18, "kb": 147.97},
}

TARGETS = ["GR", "DT", "RHOB", "VSH", "PHIE", "SWE", "PHIT"]

# ── load models ─────────────────────────────────────────────────────────────
print("Loading ML models ...")
pipelines = {}
for target in TARGETS:
    try:
        model     = joblib.load(os.path.join(models_dir, f"{target}_model.joblib"))
        scaler    = joblib.load(os.path.join(models_dir, f"{target}_scaler.joblib"))
        cal       = joblib.load(os.path.join(models_dir, f"{target}_calibrator.joblib"))
        cal_type  = joblib.load(os.path.join(models_dir, f"{target}_calibrator_type.joblib"))
        sel_idx   = joblib.load(os.path.join(models_dir, f"{target}_selected_indices.joblib"))
        pipelines[target] = {"model": model, "scaler": scaler, "calibrator": cal,
                              "cal_type": cal_type, "indices": sel_idx}
        print(f"  ✓ {target}")
    except Exception as e:
        print(f"  ✗ {target}: {e}")

# Helper for 2D pad shift
def pad_shift_2d(arr, shift):
    if shift == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if shift > 0:
        out[:, :-shift] = arr[:, shift:]
    else:
        s = -shift
        out[:, s:] = arr[:, :-s]
    return out

# ── open SEG-Y ──────────────────────────────────────────────────────────────
print("\nOpening SEG-Y ...")
f_segy = segyio.open(segy_path, "r", ignore_geometry=True)
seis_times   = f_segy.samples.astype(float)
dt_ms        = float(seis_times[1] - seis_times[0])
inlines_all  = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
xlines_all   = f_segy.attributes(segyio.TraceField.TraceNumber)[:]
n_traces     = len(inlines_all)

# Read all traces
print(f"Collecting all {n_traces:,} traces into memory ...")
all_traces = segyio.tools.collect(f_segy.trace)[:] # shape (n_traces, n_samples)
f_segy.close()

# Global AI normalisation bounds (same as training — compute on a sample)
print("Computing global AI bounds (sample) ...")
sample_step = max(1, n_traces // 500)
ai_vals = []
for ti in range(0, n_traces, sample_step):
    tr = all_traces[ti]
    ai_vals.extend(np.cumsum(np.clip(tr, -1.0, 1.0)).tolist())
global_ai_min = float(np.percentile(ai_vals, 1))
global_ai_max = float(np.percentile(ai_vals, 99))
print(f"  AI range: {global_ai_min:.3f} – {global_ai_max:.3f}")

# Reservoir window indices
res_mask = (seis_times >= RESERVOIR_T_START) & (seis_times <= RESERVOIR_T_END)
res_idx  = np.where(res_mask)[0]
print(f"Reservoir window: {RESERVOIR_T_START}–{RESERVOIR_T_END} ms  ({len(res_idx)} samples)")

# ── Vectorized Feature Generation ───────────────────────────────────────────
print("\nComputing features in parallel ...")
t_start_feat = time.time()
amp = np.nan_to_num(all_traces, nan=0.0)

# Hilbert transform
analytic = hilbert(amp, axis=1)
envelope = np.abs(analytic)
phase = np.unwrap(np.angle(analytic), axis=1)

dt_s = dt_ms / 1000.0
ifreq = np.gradient(phase, dt_s, axis=1) / (2.0 * np.pi)
ifreq = np.nan_to_num(ifreq, nan=0.0)

env_deriv = np.nan_to_num(np.gradient(envelope, axis=1), nan=0.0)

rel_pos = np.linspace(0.0, 1.0, amp.shape[1])
rel_pos_2d = np.tile(rel_pos, (n_traces, 1))

impedance = np.cumsum(np.clip(amp, -1.0, 1.0), axis=1)
impedance = np.nan_to_num(impedance, nan=0.0)
impedance = (impedance - global_ai_min) / (global_ai_max - global_ai_min + 1e-8)

max_abs_env = np.max(envelope, axis=1, keepdims=True)
eps = 1e-6 * max_abs_env
env_ratio = np.clip(envelope / (np.abs(amp) + eps), 0.0, MAX_ENV_RATIO)

# Sweetness = envelope / sqrt(max(ifreq, 0.1))
sweetness = envelope / np.sqrt(np.maximum(ifreq, 0.1))

# Rolling stats (window=5)
s0 = amp
s1 = pad_shift_2d(amp, 1)
s2 = pad_shift_2d(amp, 2)
s3 = pad_shift_2d(amp, -1)
s4 = pad_shift_2d(amp, -2)

win_max = np.maximum.reduce([s0, s1, s2, s3, s4])
win_min = np.minimum.reduce([s0, s1, s2, s3, s4])
win_mean = (s0 + s1 + s2 + s3 + s4) / 5.0
win_std = np.sqrt(((s0 - win_mean)**2 + (s1 - win_mean)**2 + (s2 - win_mean)**2 + (s3 - win_mean)**2 + (s4 - win_mean)**2) / 5.0)

attrs_dict = {
    "acoustic_impedance": impedance,
    "amp_center": amp,
    "amp_shift_+1": pad_shift_2d(amp, 1),
    "amp_shift_+2": pad_shift_2d(amp, 2),
    "amp_shift_-1": pad_shift_2d(amp, -1),
    "amp_shift_-2": pad_shift_2d(amp, -2),
    "env_center": envelope,
    "env_deriv": env_deriv,
    "env_ratio": env_ratio,
    "env_shift_+1": pad_shift_2d(envelope, 1),
    "env_shift_+2": pad_shift_2d(envelope, 2),
    "env_shift_-1": pad_shift_2d(envelope, -1),
    "env_shift_-2": pad_shift_2d(envelope, -2),
    "ifreq_center": ifreq,
    "ifreq_shift_+1": pad_shift_2d(ifreq, 1),
    "ifreq_shift_+2": pad_shift_2d(ifreq, 2),
    "ifreq_shift_-1": pad_shift_2d(ifreq, -1),
    "ifreq_shift_-2": pad_shift_2d(ifreq, -2),
    "sweetness": sweetness,
    "sweetness_shift_+1": pad_shift_2d(sweetness, 1),
    "sweetness_shift_+2": pad_shift_2d(sweetness, 2),
    "sweetness_shift_-1": pad_shift_2d(sweetness, -1),
    "sweetness_shift_-2": pad_shift_2d(sweetness, -2),
    "rel_pos": rel_pos_2d,
    "win_max": win_max,
    "win_mean": win_mean,
    "win_min": win_min,
    "win_std": win_std,
}

# Prefix features and sort alphabetically
feature_names = sorted([f"attr_{k}" for k in attrs_dict.keys()])

# Stack features: shape (n_traces, n_samples, n_features)
X_all_features = np.stack([attrs_dict[name.replace("attr_", "")] for name in feature_names], axis=2)

# Slice to reservoir window
X_res = X_all_features[:, res_idx, :] # shape (n_traces, len(res_idx), n_features)
n_features = len(feature_names)
X_res_2d = X_res.reshape(-1, n_features)

print(f"Features computed in {time.time() - t_start_feat:.1f}s. Input shape for inference: {X_res_2d.shape}")

# ── Run ML Inference ────────────────────────────────────────────────────────
print("\nRunning ML model predictions ...")
t_start_inf = time.time()
predictions_averaged = {}

for target, pipe in pipelines.items():
    t0 = time.time()
    X_sel = X_res_2d[:, pipe["indices"]]
    X_scaled = pipe["scaler"].transform(X_sel)
    pred_raw = pipe["model"].predict(X_scaled)
    
    # Bypass calibration - work with raw model predictions directly
    pred_cal = pred_raw
        
    # Reshape back and average along reservoir window axis
    pred_cal_2d = pred_cal.reshape(n_traces, len(res_idx))
    pred_mean = np.mean(pred_cal_2d, axis=1)
    predictions_averaged[target] = pred_mean
    print(f"  ✓ {target} predicted in {time.time() - t0:.1f}s")

# ── Assemble Grid Data ──────────────────────────────────────────────────────
print("\nAssembling grid points ...")
grid_points = []
for ti in range(n_traces):
    il = int(inlines_all[ti])
    xl = int(xlines_all[ti])
    point = {"il": il, "xl": xl}
    for target in TARGETS:
        point[target] = round(float(predictions_averaged[target][ti]), 4)
        
    # Sweet Spot Index
    phi = point.get("PHIT", point.get("PHIE", 0))
    vsh = point.get("VSH", 0)
    swe = point.get("SWE", 1)
    point["SSI"] = round(float(phi * max(0, 1 - vsh) * max(0, 1 - swe)), 5)
    grid_points.append(point)

print(f"Inference and assembly completed in {time.time() - t_start_inf:.1f}s")

# ── well inline/crossline from tie summary ──────────────────────────────────
import csv
tie_csv = os.path.join(analysis_dir, "well_seismic", "output", "tie_summary.csv")
well_grid = {}
try:
    with open(tie_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            wname = row["well"]
            well_grid[wname] = {
                "name": wname,
                "il":   int(float(row["inline"])),
                "xl":   int(float(row["crossline"])),
                "x":    WELLS[wname]["x"],
                "y":    WELLS[wname]["y"],
            }
except Exception as e:
    print(f"Warning: could not read tie CSV: {e}")

# ── export JS ───────────────────────────────────────────────────────────────
print(f"\nExporting {len(grid_points):,} grid points to {output_js} ...")
all_inlines   = [int(x) for x in sorted(list(set(inlines_all.astype(int))))]
all_crosslines = [int(x) for x in sorted(list(set(xlines_all.astype(int))))]

export = {
    "inlines":    all_inlines,
    "crosslines": all_crosslines,
    "targets":    TARGETS + ["SSI"],
    "reservoir":  {"t_start": RESERVOIR_T_START, "t_end": RESERVOIR_T_END},
    "wells":      list(well_grid.values()),
    "points":     grid_points,
}

js_content = f"export const gridData = {json.dumps(export, separators=(',', ':'))};\n"

with open(output_js, "w", encoding="utf-8") as fh:
    fh.write(js_content)

size_mb = os.path.getsize(output_js) / 1e6
print(f"✓ Written {output_js}  ({size_mb:.1f} MB)")
