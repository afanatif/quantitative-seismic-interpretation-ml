import os
import re
import json
import numpy as np
import pandas as pd
import segyio
import lasio
import joblib
from scipy.signal import hilbert

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
segy_path = os.path.join(analysis_dir, "segy", "origional.segy")
las_dir = os.path.join(analysis_dir, "las")
tie_summary_path = os.path.join(analysis_dir, "well_seismic", "output", "tie_summary.csv")
models_dir = os.path.join(analysis_dir, "ml_outputs_v3", "models")
output_js_path = os.path.join(analysis_dir, "frontend", "src", "data.js")

FEET_TO_METERS = 0.3048
MAX_ENV_RATIO = 50.0

WELLS = {
    "Z-02": {"x": 1205859.09, "y": 9692966.31, "kb": 146.46},
    "Z-03": {"x": 1201178.25, "y": 9682452.00, "kb": 147.64},
    "Z-04": {"x": 1205820.18, "y": 9696292.65, "kb": 147.64},
    "Z-05": {"x": 1206404.17, "y": 9679510.83, "kb": 144.36},
    "Z-06": {"x": 1207337.37, "y": 9684145.64, "kb": 146.98},
    "Z-07": {"x": 1206364.34, "y": 9688320.18, "kb": 147.97}
}

target_curves = ["GR", "DT", "RHOB", "VSH", "PHIE", "SWE", "PHIT"]

def _pad_shift(arr: np.ndarray, shift: int) -> np.ndarray:
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

def shift_trace(arr: np.ndarray, shift_ms: float, dt_ms: float) -> np.ndarray:
    shift_samples = int(round(shift_ms / dt_ms))
    if shift_samples == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if shift_samples > 0:
        out[shift_samples:] = arr[:-shift_samples]
    else:
        out[:shift_samples] = arr[-shift_samples:]
    return out

def _rolling_stats(arr: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    half = window // 2
    maxv = np.zeros(n)
    meanv = np.zeros(n)
    minv = np.zeros(n)
    stdv = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = arr[lo:hi]
        maxv[i] = float(np.max(window_vals))
        meanv[i] = float(np.mean(window_vals))
        minv[i] = float(np.min(window_vals))
        stdv[i] = float(np.std(window_vals))
    return maxv, meanv, minv, stdv

def compute_trace_attributes(trace, times_ms, dt_ms, global_ai_min, global_ai_max) -> dict[str, np.ndarray]:
    trace = np.asarray(trace, dtype=float)
    n_samples = len(trace)
    amp = np.nan_to_num(trace, nan=0.0)
    analytic = hilbert(amp)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dt_s = dt_ms / 1000.0
    ifreq = np.gradient(phase, dt_s) / (2.0 * np.pi)
    ifreq = np.nan_to_num(ifreq, nan=0.0)
    rel_pos = np.linspace(0.0, 1.0, n_samples)
    env_deriv = np.nan_to_num(np.gradient(envelope), nan=0.0)
    
    impedance = np.cumsum(np.clip(amp, -1.0, 1.0))
    impedance = np.nan_to_num(impedance, nan=0.0)
    impedance = (impedance - global_ai_min) / (global_ai_max - global_ai_min + 1e-8)

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
    return {f"attr_{k}": v for k, v in attrs.items()}

def ricker_wavelet(freq_hz, length_ms, dt_ms):
    t = np.arange(-length_ms / 2, length_ms / 2 + dt_ms, dt_ms) / 1000.0
    y = (1 - 2 * (np.pi * freq_hz * t) ** 2) * np.exp(-(np.pi * freq_hz * t) ** 2)
    return y

def rotate_phase(wavelet, phase_deg):
    analytic = hilbert(wavelet)
    phase_rad = np.deg2rad(phase_deg)
    rotated = np.real(analytic * np.exp(1j * phase_rad))
    return rotated

def main():
    print("Generating comprehensive frontend dataset...")
    df_ties = pd.read_csv(tie_summary_path)
    
    # Load ML models and their selected indices/scalers
    pipelines = {}
    for target in target_curves:
        selected_indices = joblib.load(os.path.join(models_dir, f"{target}_selected_indices.joblib"))
        scaler = joblib.load(os.path.join(models_dir, f"{target}_scaler.joblib"))
        model = joblib.load(os.path.join(models_dir, f"{target}_model.joblib"))
        calibrator = joblib.load(os.path.join(models_dir, f"{target}_calibrator.joblib"))
        cal_type = joblib.load(os.path.join(models_dir, f"{target}_calibrator_type.joblib"))
        
        pipelines[target] = {
            "indices": selected_indices,
            "scaler": scaler,
            "model": model,
            "calibrator": calibrator,
            "cal_type": cal_type
        }
    
    # Define attribute order to match ML model features
    df_train_tbl = pd.read_csv(os.path.join(analysis_dir, "training_data", "output", "training_table.csv"))
    attr_cols = [c for c in df_train_tbl.columns if c.startswith("attr_")]
    
    # Get global acoustic impedance normalization bounds from training table
    global_ai_min = -11.0
    global_ai_max = 14.0

    wells_out = {}

    with segyio.open(segy_path, "r", ignore_geometry=True) as f_segy:
        seis_times = f_segy.samples.astype(float)
        dt_ms = float(seis_times[1] - seis_times[0])
        inlines = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
        crosslines = f_segy.attributes(segyio.TraceField.TraceNumber)[:]

        for _, row in df_ties.iterrows():
            well_name = row["well"]
            shift_ms = row["shift_ms"]
            corr = row["correlation"]
            mapped_inline = int(row["inline"])
            mapped_crossline = int(row["crossline"])
            freq_hz = float(row["freq_hz"])
            phase_deg = float(row["phase_deg"])

            trace_idx = np.where((inlines == mapped_inline) & (crosslines == mapped_crossline))[0]
            if len(trace_idx) == 0:
                continue
            trace_idx = int(trace_idx[0])

            raw_trace = f_segy.trace[trace_idx].copy()
            polarity = -1.0 if corr < 0 else 1.0
            trace = raw_trace * polarity

            # Build trace attributes along the entire trace
            attrs_dict = compute_trace_attributes(trace, seis_times, dt_ms, global_ai_min, global_ai_max)
            
            # Form feature matrix X matching columns order
            X_raw = np.column_stack([attrs_dict[c] for c in attr_cols])

            # Pre-compute predictions for all targets
            preds = {}
            preds_raw = {}
            for target, pipe in pipelines.items():
                X_sel = X_raw[:, pipe["indices"]]
                X_scaled = pipe["scaler"].transform(X_sel)
                pred_raw = pipe["model"].predict(X_scaled)
                
                # Bypass calibration - export raw model predictions directly
                pred_cal = pred_raw
                preds[target] = pred_cal
                preds_raw[target] = pred_raw

            # Load actual logs from LAS file
            las_path = os.path.join(las_dir, f"{well_name}.las")
            las = lasio.read(las_path)
            df_las = las.df()

            dptm_ms = df_las["DPTM"].values.astype(float) if "DPTM" in df_las.columns else None
            well_times_mapped = dptm_ms + shift_ms if dptm_ms is not None else None

            # Calculate raw unshifted reflectivity curve from DT and RHOB
            refl_reg_raw = np.zeros(len(seis_times))
            if "DT" in df_las.columns and "RHOB" in df_las.columns and dptm_ms is not None:
                # Vp in m/s
                vp_ms = (1.0e6 / df_las["DT"].values.astype(float)) * 0.3048
                impedance = vp_ms * df_las["RHOB"].values.astype(float)
                # RC
                refl = (impedance[1:] - impedance[:-1]) / (impedance[1:] + impedance[:-1])
                refl_time_raw = (dptm_ms[1:] + dptm_ms[:-1]) / 2.0
                
                # Resample raw unshifted reflectivity to seismic times
                t_start = np.ceil(refl_time_raw.min() / dt_ms) * dt_ms
                bin_idx = np.clip(((refl_time_raw - seis_times[0]) / dt_ms).round().astype(int), 0, len(seis_times) - 1)
                counts = np.zeros(len(seis_times))
                for b, r in zip(bin_idx, refl):
                    refl_reg_raw[b] += r
                    counts[b] += 1
                counts[counts == 0] = 1
                refl_reg_raw = refl_reg_raw / counts
                
            # Pre-generate synthetic seismogram using SHIFTED reflectivity
            refl_shifted = shift_trace(refl_reg_raw, shift_ms, dt_ms)
            wav = ricker_wavelet(freq_hz, 100, dt_ms)
            wav_rotated = rotate_phase(wav, phase_deg)
            synthetic = np.convolve(refl_shifted, wav_rotated, mode="same")
            
            # Scale synthetic to match seismic RMS
            rms_seis = np.std(trace)
            rms_syn = np.std(synthetic)
            scale = rms_syn / rms_seis if rms_seis != 0 else 1.0
            synthetic = synthetic / scale if scale != 0 else synthetic

            # For each seismic time sample, align the actual logs using block averaging
            well_samples = []
            for i, t in enumerate(seis_times):
                sample_data = {
                    "time": float(t),
                    "seismic_amp": float(trace[i]),
                    "synthetic_amp": float(synthetic[i]),
                    "reflectivity": float(refl_reg_raw[i]) # export unshifted so simulator works interactively
                }
                
                # Target actual logs
                if well_times_mapped is not None:
                    idx_in_bin = np.where((well_times_mapped >= t - dt_ms/2) & (well_times_mapped < t + dt_ms/2))[0]
                    if len(idx_in_bin) > 0:
                        for target in target_curves:
                            col_name = "PHIE" if target == "PHIE" else target
                            if col_name in df_las.columns:
                                val = float(np.mean(df_las[col_name].values[idx_in_bin]))
                                sample_data[f"{target} (Act)"] = None if np.isnan(val) else val
                    else:
                        for target in target_curves:
                            sample_data[f"{target} (Act)"] = None

                # ML Predicted logs
                for target in target_curves:
                    sample_data[f"{target} (Pred)"] = float(preds[target][i])
                    sample_data[f"{target} (Pred Raw)"] = float(preds_raw[target][i])
                
                well_samples.append(sample_data)

            # Keep a wider seismic window to allow time shift sliding (-30ms to +30ms)
            # and convolution (length 51 samples = 100ms) to compute exact correlation on the logged interval
            if well_times_mapped is not None:
                start_t = well_times_mapped.min() - 80
                end_t = well_times_mapped.max() + 150
                well_samples = [s for s in well_samples if s["time"] >= start_t and s["time"] <= end_t]

            wells_out[well_name] = {
                "name": well_name,
                "x": WELLS[well_name]["x"],
                "y": WELLS[well_name]["y"],
                "kb": WELLS[well_name]["kb"],
                "inline": mapped_inline,
                "crossline": mapped_crossline,
                "tie": {
                    "correlation": float(corr),
                    "shift_ms": float(shift_ms),
                    "freq_hz": float(freq_hz),
                    "phase_deg": float(phase_deg),
                    "quality": str(row["quality"])
                },
                "samples": well_samples
            }

    # Write JS file
    output_js_content = f"""// Automatic generated database for LMKR Geologist Dashboard
export const blindWellName = "Z-04";

export const wellsConfig = {json.dumps(WELLS, indent=2)};

export const wellsData = {json.dumps(wells_out, indent=2)};
"""

    with open(output_js_path, "w", encoding="utf-8") as f:
        f.write(output_js_content)
    
    print(f"Successfully generated {output_js_path} for all 6 wells!")

if __name__ == "__main__":
    main()
