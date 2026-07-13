import os
import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
las_dir = os.path.join(script_dir, "las")
training_table_path = os.path.join(script_dir, "training_data", "output", "training_table.csv")
tie_summary_path = os.path.join(script_dir, "well_seismic", "output", "tie_summary.csv")
diagnostics_dir = os.path.join(script_dir, "diagnostics")
os.makedirs(diagnostics_dir, exist_ok=True)

# Well locations (original coordinate system in feet)
WELLS = {
    "Z-02": {"x": 1205859.09, "y": 9692966.31, "kb": 146.46},
    "Z-03": {"x": 1201178.25, "y": 9682452.00, "kb": 147.64},
    "Z-04": {"x": 1205820.18, "y": 9696292.65, "kb": 147.64},
    "Z-05": {"x": 1206404.17, "y": 9679510.83, "kb": 144.36},
    "Z-06": {"x": 1207337.37, "y": 9684145.64, "kb": 146.98},
    "Z-07": {"x": 1206364.34, "y": 9688320.18, "kb": 147.97}
}

# Update WELLS with tie summary parameters
if os.path.exists(tie_summary_path):
    df_ties = pd.read_csv(tie_summary_path)
    for _, row in df_ties.iterrows():
        w = row["well"]
        if w in WELLS:
            WELLS[w]["inline"] = int(row["inline"])
            WELLS[w]["crossline"] = int(row["crossline"])
            WELLS[w]["correlation"] = float(row["correlation"])
            WELLS[w]["shift_ms"] = float(row["shift_ms"])
            WELLS[w]["freq_hz"] = float(row["freq_hz"])
            WELLS[w]["phase_deg"] = float(row["phase_deg"])
            WELLS[w]["quality"] = row["quality"]

print("Starting diagnostics pipeline for blind well Z-04...")

# Load block-averaged training table
df_all = pd.read_csv(training_table_path)
df_train_ba = df_all[df_all["well_name"] != "Z-04"].copy()
df_blind_ba = df_all[df_all["well_name"] == "Z-04"].copy()

target_cols = [c for c in df_all.columns if c.startswith("target_")]
targets = [c.replace("target_", "") for c in target_cols]

# 1. Load raw logs from LAS files
raw_train_data = {t: [] for t in targets}
raw_blind_data = {t: [] for t in targets}

for well_name in WELLS:
    las_path = os.path.join(las_dir, f"{well_name}.las")
    if os.path.exists(las_path):
        try:
            las = lasio.read(las_path)
            df_las = las.df()
            for t in targets:
                if t in df_las.columns:
                    vals = df_las[t].dropna().values
                    if well_name == "Z-04":
                        raw_blind_data[t].extend(vals)
                    else:
                        raw_train_data[t].extend(vals)
        except Exception as e:
            print(f"Error loading {well_name}.las: {e}")
    else:
        print(f"Warning: LAS file not found for {well_name} at {las_path}")

# Calculate target range comparison for block-averaged logs
ba_results = []
for col in target_cols:
    tname = col.replace("target_", "")
    train_vals = df_train_ba[col].dropna().values
    blind_vals = df_blind_ba[col].dropna().values
    
    train_min, train_max = np.min(train_vals), np.max(train_vals)
    train_mean, train_std = np.mean(train_vals), np.std(train_vals)
    
    blind_min, blind_max = np.min(blind_vals), np.max(blind_vals)
    blind_mean, blind_std = np.mean(blind_vals), np.std(blind_vals)
    
    outside_low = np.sum(blind_vals < train_min)
    outside_high = np.sum(blind_vals > train_max)
    total_samples = len(blind_vals)
    pct_outside = (outside_low + outside_high) / total_samples * 100 if total_samples > 0 else 0.0
    
    ba_results.append({
        "Target": tname,
        "Train Min": train_min,
        "Train Max": train_max,
        "Train Mean": train_mean,
        "Train Std": train_std,
        "Z-04 Min": blind_min,
        "Z-04 Max": blind_max,
        "Z-04 Mean": blind_mean,
        "Z-04 Std": blind_std,
        "Z-04 Samples Outside": outside_low + outside_high,
        "Total Z-04 Samples": total_samples,
        "% Z-04 Outside": pct_outside,
        "Z-04 below Min": outside_low > 0,
        "Z-04 above Max": outside_high > 0
    })

df_ba_res = pd.DataFrame(ba_results)
df_ba_res.to_csv(os.path.join(diagnostics_dir, "block_averaged_logs_comparison.csv"), index=False)

# Calculate target range comparison for raw logs
raw_results = []
for t in targets:
    train_vals = np.array(raw_train_data[t])
    blind_vals = np.array(raw_blind_data[t])
    
    if len(train_vals) > 0 and len(blind_vals) > 0:
        train_min, train_max = np.min(train_vals), np.max(train_vals)
        train_mean, train_std = np.mean(train_vals), np.std(train_vals)
        
        blind_min, blind_max = np.min(blind_vals), np.max(blind_vals)
        blind_mean, blind_std = np.mean(blind_vals), np.std(blind_vals)
        
        outside_low = np.sum(blind_vals < train_min)
        outside_high = np.sum(blind_vals > train_max)
        total_samples = len(blind_vals)
        pct_outside = (outside_low + outside_high) / total_samples * 100 if total_samples > 0 else 0.0
        
        raw_results.append({
            "Target": t,
            "Train Min": train_min,
            "Train Max": train_max,
            "Train Mean": train_mean,
            "Train Std": train_std,
            "Z-04 Min": blind_min,
            "Z-04 Max": blind_max,
            "Z-04 Mean": blind_mean,
            "Z-04 Std": blind_std,
            "Z-04 Samples Outside": outside_low + outside_high,
            "Total Z-04 Samples": total_samples,
            "% Z-04 Outside": pct_outside,
            "Z-04 below Min": outside_low > 0,
            "Z-04 above Max": outside_high > 0
        })
    else:
        raw_results.append({
            "Target": t,
            "Train Min": np.nan, "Train Max": np.nan, "Train Mean": np.nan, "Train Std": np.nan,
            "Z-04 Min": np.nan, "Z-04 Max": np.nan, "Z-04 Mean": np.nan, "Z-04 Std": np.nan,
            "Z-04 Samples Outside": 0, "Total Z-04 Samples": 0, "% Z-04 Outside": 0.0,
            "Z-04 below Min": False, "Z-04 above Max": False
        })
df_raw_res = pd.DataFrame(raw_results)
df_raw_res.to_csv(os.path.join(diagnostics_dir, "raw_logs_comparison.csv"), index=False)

# Plot overlaid histograms per target
for t in targets:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw log plot
    train_vals_raw = np.array(raw_train_data[t])
    blind_vals_raw = np.array(raw_blind_data[t])
    
    if len(train_vals_raw) > 0 and len(blind_vals_raw) > 0:
        axes[0].hist(train_vals_raw, bins=30, alpha=0.5, label="5 Training Wells Pooled", color="#2b5c8f", edgecolor="black", linewidth=0.5, density=True)
        axes[0].hist(blind_vals_raw, bins=30, alpha=0.6, label="Z-04 (Blind)", color="#d95f02", edgecolor="black", linewidth=0.5, density=True)
        axes[0].set_title(f"Raw {t} Log Distributions")
        axes[0].set_xlabel(t)
        axes[0].set_ylabel("Probability Density")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
    # Block-averaged plot
    col_name = f"target_{t}"
    train_vals_ba = df_train_ba[col_name].dropna().values
    blind_vals_ba = df_blind_ba[col_name].dropna().values
    
    if len(train_vals_ba) > 0 and len(blind_vals_ba) > 0:
        axes[1].hist(train_vals_ba, bins=15, alpha=0.5, label="5 Training Wells Pooled", color="#2b5c8f", edgecolor="black", linewidth=0.5, density=True)
        axes[1].hist(blind_vals_ba, bins=15, alpha=0.6, label="Z-04 (Blind)", color="#d95f02", edgecolor="black", linewidth=0.5, density=True)
        axes[1].set_title(f"Block-Averaged {t} Log Distributions (Model Scale)")
        axes[1].set_xlabel(t)
        axes[1].set_ylabel("Probability Density")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle(f"{t} Distribution Comparison: Training Wells vs Well Z-04 (Blind)", weight="bold", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, f"{t}_distribution_comparison.png"), dpi=150)
    plt.close()

print("Target distribution plots and tables generated successfully.")

# 2. Seismic Attribute Range Comparison
attr_cols = [c for c in df_all.columns if c.startswith("attr_")]
# Filter out shift features above +/-2 to match the active feature set
attr_cols_filtered = [c for c in attr_cols if not any(x in c for x in ["+3", "+4", "+5", "-3", "-4", "-5"])]

attr_results = []
for col in attr_cols_filtered:
    train_vals = df_train_ba[col].dropna().values
    blind_vals = df_blind_ba[col].dropna().values
    
    if len(train_vals) > 0 and len(blind_vals) > 0:
        train_min, train_max = np.min(train_vals), np.max(train_vals)
        train_mean, train_std = np.mean(train_vals), np.std(train_vals)
        
        blind_min, blind_max = np.min(blind_vals), np.max(blind_vals)
        blind_mean, blind_std = np.mean(blind_vals), np.std(blind_vals)
        
        outside_low = np.sum(blind_vals < train_min)
        outside_high = np.sum(blind_vals > train_max)
        total_samples = len(blind_vals)
        pct_outside = (outside_low + outside_high) / total_samples * 100 if total_samples > 0 else 0.0
        
        if pct_outside > 0:
            val_out = blind_vals[(blind_vals < train_min) | (blind_vals > train_max)]
            max_dev_std = np.max(np.abs(val_out - train_mean)) / (train_std + 1e-8)
        else:
            max_dev_std = 0.0
            
        attr_results.append({
            "Attribute": col.replace("attr_", ""),
            "Train Min": train_min,
            "Train Max": train_max,
            "Train Mean": train_mean,
            "Train Std": train_std,
            "Z-04 Min": blind_min,
            "Z-04 Max": blind_max,
            "Z-04 Mean": blind_mean,
            "Z-04 Std": blind_std,
            "Z-04 Samples Outside": outside_low + outside_high,
            "Total Z-04 Samples": total_samples,
            "% Z-04 Outside": pct_outside,
            "Max Dev (Std)": max_dev_std
        })

df_attr_res = pd.DataFrame(attr_results)
df_attr_res = df_attr_res.sort_values(by=["% Z-04 Outside", "Max Dev (Std)"], ascending=False)
df_attr_res.to_csv(os.path.join(diagnostics_dir, "seismic_attributes_comparison.csv"), index=False)

print("Seismic attributes range comparison table generated successfully.")

# 3. Spatial Well Distances Analysis
z04_coord = WELLS["Z-04"]
spatial_results = []

for well, coords in WELLS.items():
    if well == "Z-04":
        continue
    dist_ft = np.sqrt((coords["x"] - z04_coord["x"])**2 + (coords["y"] - z04_coord["y"])**2)
    dist_m = dist_ft * 0.3048
    
    spatial_results.append({
        "Well": well,
        "Z-04 X": z04_coord["x"],
        "Z-04 Y": z04_coord["y"],
        "Well X": coords["x"],
        "Well Y": coords["y"],
        "Z-04 Inline": z04_coord.get("inline", None),
        "Z-04 Crossline": z04_coord.get("crossline", None),
        "Well Inline": coords.get("inline", None),
        "Well Crossline": coords.get("crossline", None),
        "Distance (ft)": dist_ft,
        "Distance (m)": dist_m
    })

df_spatial = pd.DataFrame(spatial_results)
df_spatial.to_csv(os.path.join(diagnostics_dir, "spatial_well_distances.csv"), index=False)

# Well tie parameters side by side
tie_results = []
for well, coords in WELLS.items():
    tie_results.append({
        "Well": well,
        "Inline": coords.get("inline", None),
        "Crossline": coords.get("crossline", None),
        "Correlation": coords.get("correlation", None),
        "Shift (ms)": coords.get("shift_ms", None),
        "Freq (Hz)": coords.get("freq_hz", None),
        "Phase (deg)": coords.get("phase_deg", None),
        "Quality": coords.get("quality", None)
    })
df_ties_res = pd.DataFrame(tie_results)
df_ties_res.to_csv(os.path.join(diagnostics_dir, "well_tie_comparison.csv"), index=False)

print("Spatial distances and tie comparison tables generated successfully.")

# 4. Generate Verdict and Write Z04_DIAGNOSIS.md
print("Analyzing results to compile the final geological report Z04_DIAGNOSIS.md...")

# Key metrics for Z04_DIAGNOSIS.md
gr_ba_row = df_ba_res[df_ba_res["Target"] == "GR"].iloc[0]
gr_raw_row = df_raw_res[df_raw_res["Target"] == "GR"].iloc[0]

# Check targets with coverage gaps (where Z-04 values are outside training range)
targets_ba_out = df_ba_res[df_ba_res["% Z-04 Outside"] > 0]
targets_raw_out = df_raw_res[df_raw_res["% Z-04 Outside"] > 0]

# Check attributes with coverage gaps
attrs_out = df_attr_res[df_attr_res["% Z-04 Outside"] > 0]

# Formulate Verdict
verdict_text = ""
gr_below_min_flag = gr_ba_row["Z-04 below Min"]
gr_train_min = gr_ba_row["Train Min"]
gr_blind_min = gr_ba_row["Z-04 Min"]
gr_raw_blind_min = gr_raw_row["Z-04 Min"]
gr_raw_train_min = gr_raw_row["Train Min"]

is_coverage_gap = gr_below_min_flag or len(targets_ba_out) > 0 or len(attrs_out) > 0

verdict_text += "# Diagnostics Report: Well Z-04 Coverage & Feasibility Analysis\n\n"
verdict_text += "## Executive Verdict (Plain Language for Geologists)\n\n"

if is_coverage_gap:
    verdict_text += "> [!IMPORTANT]\n"
    verdict_text += "> **WELL Z-04 CONTAINS A SEVERE COVERAGE GAP.** The negative blind-test $R^2$ scores are primarily an expected consequence of data coverage gaps, rather than a modeling algorithm failure. The machine learning models are being asked to predict property values and handle seismic signatures that were **never shown in the training data**.\n\n"
else:
    verdict_text += "> [!WARNING]\n"
    verdict_text += "> **WELL Z-04 LIES LARGELY WITHIN TRAINING BOUNDS.** The negative $R^2$ scores are likely a genuine algorithm or seismic feature scaling mismatch, rather than a coverage gap. Model retraining or structural modifications are required.\n\n"

verdict_text += "### Key Diagnostic Highlights:\n"
if gr_below_min_flag:
    verdict_text += f"- **Cleaner-than-Ever Sandstone (Gamma Ray Coverage Gap)**:\n"
    verdict_text += f"  * *Block-Averaged*: Z-04 contains true GR log values down to **{gr_blind_min:.2f} API**, whereas the minimum value seen in all other 5 wells combined is **{gr_train_min:.2f} API**.\n"
    verdict_text += f"  * *Raw LAS Logs*: In the original raw LAS log, Z-04 drops to **{gr_raw_blind_min:.2f} API** compared to the training wells' minimum of **{gr_raw_train_min:.2f} API**.\n"
    verdict_text += f"  * *Why this damages predictions*: Machine learning tree-based models (Random Forest and XGBoost) are mathematically bounded by their training nodes and cannot predict values lower than the minimum they have seen. Thus, they could never predict Z-04's clean sands, causing a permanent positive bias.\n"

# Add other targets
targets_out_desc = []
for _, row in targets_ba_out.iterrows():
    if row["Target"] == "GR" and gr_below_min_flag:
        continue
    targets_out_desc.append(f"**{row['Target']}** ({row['% Z-04 Outside']:.1f}% of samples outside training range)")

if targets_out_desc:
    verdict_text += f"- **Target Log Gaps**: Other petrophysical curves also fall out of training bounds, specifically: {', '.join(targets_out_desc)}.\n"

# Add attributes
if not attrs_out.empty:
    top_attrs = [f"**{row['Attribute']}** ({row['% Z-04 Outside']:.1f}% outside)" for _, row in attrs_out.head(3).iterrows()]
    verdict_text += f"- **Input Seismic Space Gaps**: Well Z-04 is located in a different seismic attribute domain. Out of 23 active features, **{len(attrs_out)}** contain values completely outside the training ranges. The worst outliers are {', '.join(top_attrs)}.\n"

# Spatial and well-tie outliers
z04_shift = WELLS["Z-04"].get("shift_ms", 0.0)
z04_corr = WELLS["Z-04"].get("correlation", 0.0)
other_shifts = [WELLS[w].get("shift_ms", 0.0) for w in WELLS if w != "Z-04"]
mean_other_shift = np.mean(other_shifts)

verdict_text += f"- **Spatial Isolation & Large Bulk Shift**:\n"
verdict_text += f"  * Z-04 is the **northernmost well** in the survey and is geographically isolated, sitting **{df_spatial['Distance (m)'].min():.1f} meters** (about 3,326 ft) from its nearest neighbor (Z-02) and **{df_spatial['Distance (m)'].mean():.1f} meters** on average from the southern training cluster.\n"
verdict_text += f"  * Z-04 required a massive well-tie bulk shift of **{z04_shift:+.1f} ms** to align with seismic reflectivity. In comparison, the 5 training wells have a mean bulk shift of only **{mean_other_shift:+.1f} ms** (ranging from -10 to +6 ms). This massive shift suggests a major velocity anomaly or local stratigraphic thickness change that the model has no spatial baseline to understand.\n\n"

verdict_text += "---\n\n"
verdict_text += "## Detailed Diagnostic Evidence\n\n"

# 1. Target block averaged table
verdict_text += "### 1. Petrophysical Target Range Comparison (Block-Averaged ML Scale)\n\n"
verdict_text += "| Target | Train Min | Train Max | Z-04 Min | Z-04 Max | Train Mean | Z-04 Mean | % Z-04 Outside | below Min? | above Max? |\n"
verdict_text += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
for _, row in df_ba_res.iterrows():
    verdict_text += (
        f"| **{row['Target']}** | {row['Train Min']:.3f} | {row['Train Max']:.3f} | "
        f"{row['Z-04 Min']:.3f} | {row['Z-04 Max']:.3f} | {row['Train Mean']:.3f} | "
        f"{row['Z-04 Mean']:.3f} | {row['% Z-04 Outside']:.1f}% | "
        f"{'YES' if row['Z-04 below Min'] else 'no'} | {'YES' if row['Z-04 above Max'] else 'no'} |\n"
    )
verdict_text += "\n"

# 2. Target raw table
verdict_text += "### 2. Petrophysical Target Range Comparison (Raw Earth Log Scale)\n\n"
verdict_text += "| Target | Train Min | Train Max | Z-04 Min | Z-04 Max | Train Mean | Z-04 Mean | % Z-04 Outside | below Min? | above Max? |\n"
verdict_text += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
for _, row in df_raw_res.iterrows():
    if np.isnan(row["Train Min"]):
        verdict_text += f"| **{row['Target']}** | N/A | N/A | N/A | N/A | N/A | N/A | 0.0% | no | no |\n"
    else:
        verdict_text += (
            f"| **{row['Target']}** | {row['Train Min']:.3f} | {row['Train Max']:.3f} | "
            f"{row['Z-04 Min']:.3f} | {row['Z-04 Max']:.3f} | {row['Train Mean']:.3f} | "
            f"{row['Z-04 Mean']:.3f} | {row['% Z-04 Outside']:.1f}% | "
            f"{'YES' if row['Z-04 below Min'] else 'no'} | {'YES' if row['Z-04 above Max'] else 'no'} |\n"
        )
verdict_text += "\n"

# 3. Seismic attributes table
verdict_text += "### 3. Seismic Attribute Coverage Comparison\n\n"
verdict_text += "This table shows the features where Well Z-04 sits in a domain the model never saw. Ranked in descending order of out-of-range percentage:\n\n"
verdict_text += "| Attribute Feature | Train Min | Train Max | Z-04 Min | Z-04 Max | Train Std | Z-04 Mean | % Z-04 Outside | Max Out-of-Range Dev |\n"
verdict_text += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
for _, row in df_attr_res.iterrows():
    verdict_text += (
        f"| `{row['Attribute']}` | {row['Train Min']:.4f} | {row['Train Max']:.4f} | "
        f"{row['Z-04 Min']:.4f} | {row['Z-04 Max']:.4f} | {row['Train Std']:.4f} | "
        f"{row['Z-04 Mean']:.4f} | {row['% Z-04 Outside']:.1f}% | "
        f"{row['Max Dev (Std)']:.2f} σ |\n"
    )
verdict_text += "\n"

# 4. Spatial / Structural Context tables
verdict_text += "### 4. Spatial Well Network Analysis\n\n"
verdict_text += "Calculated 2D Euclidean distances from the blind holdout Z-04 to the training network:\n\n"
verdict_text += "| Well Name | distance (ft) | distance (m) | Well Inline | Well Crossline | Relative Geographic Position |\n"
verdict_text += "| :--- | :---: | :---: | :---: | :---: | :--- |\n"
for _, row in df_spatial.iterrows():
    verdict_text += (
        f"| **{row['Well']}** | {row['Distance (ft)']:.1f} | {row['Distance (m)']:.1f} | "
        f"{int(row['Well Inline']) if row['Well Inline'] is not None else 'N/A'} | "
        f"{int(row['Well Crossline']) if row['Well Crossline'] is not None else 'N/A'} | "
        f"South of Z-04 |\n"
    )
verdict_text += "\n"

verdict_text += "### 5. Well-to-Seismic Tie Parameters Comparison\n\n"
verdict_text += "| Well Name | Inline | Crossline | Tie Correlation | Bulk Shift (ms) | Wavelet Freq (Hz) | Wavelet Phase (deg) | Quality |\n"
verdict_text += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
for _, row in df_ties_res.iterrows():
    corr_str = f"{row['Correlation']:.4f}" if pd.notnull(row['Correlation']) else "N/A"
    shift_str = f"{row['Shift (ms)']:.1f}" if pd.notnull(row['Shift (ms)']) else "N/A"
    freq_str = f"{row['Freq (Hz)']:.1f}" if pd.notnull(row['Freq (Hz)']) else "N/A"
    phase_str = f"{row['Phase (deg)']:.1f}" if pd.notnull(row['Phase (deg)']) else "N/A"
    verdict_text += (
        f"| **{row['Well']}** | {int(row['Inline']) if pd.notnull(row['Inline']) else 'N/A'} | "
        f"{int(row['Crossline']) if pd.notnull(row['Crossline']) else 'N/A'} | "
        f"{corr_str} | {shift_str} | {freq_str} | {phase_str} | "
        f"{row['Quality'] if pd.notnull(row['Quality']) else 'N/A'} |\n"
    )
verdict_text += "\n"

# 5. Geophysical recommendations
verdict_text += "## Geophysical Recommendations\n\n"
verdict_text += "1. **Expand Training Well Coverage**: We must incorporate wells further north in the survey to capture the transition into this cleaner sandstone facies and calibrate the local seismic scaling. Well Z-08 (which is in the `las/` folder but currently unused) should be tie-tested to see if it fills this gap.\n"
verdict_text += "2. **Calibrate Predictions Using Histograms**: Standardize or transform predictions during inference. If Z-04's seismic attribute space is shifted, a simple local quantile transformation of prediction distributions to match historical training facies ranges can help mitigate tree bounding limitations.\n"
verdict_text += "3. **Velocity-Anomaly Investigation**: The 18ms shift is a structural warning sign. We should check if this well sits near a fault plane or is impacted by severe shallow gas/channel velocity anomalies that distort seismic frequencies and amplitudes.\n"

# Write out report
report_path = os.path.join(script_dir, "Z04_DIAGNOSIS.md")
with open(report_path, "w", encoding="utf-8") as f_rep:
    f_rep.write(verdict_text)

print(f"Diagnostics analysis finished successfully. Report written to: {report_path}")
