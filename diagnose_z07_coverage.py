import os
import numpy as np
import pandas as pd

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
training_table_path = os.path.join(script_dir, "training_data", "output", "training_table.csv")
diagnostics_dir = os.path.join(script_dir, "diagnostics")
os.makedirs(diagnostics_dir, exist_ok=True)

# Load data
df = pd.read_csv(training_table_path)

# Separate into groups
df_others = df[~df["well_name"].isin(["Z-04", "Z-07"])].copy()
df_z07 = df[df["well_name"] == "Z-07"].copy()
df_z04 = df[df["well_name"] == "Z-04"].copy()

print(f"Comparing Z-07 vs Z-04 coverage. Samples count:")
print(f"  - 4 Training Wells (Z-02, Z-03, Z-05, Z-06): {len(df_others)}")
print(f"  - Well Z-07 (Training Set):                  {len(df_z07)}")
print(f"  - Well Z-04 (Blind Test Set):                {len(df_z04)}")

# 1. Targets Comparison
target_cols = [c for c in df.columns if c.startswith("target_")]
target_res = []

print("\n--- Target Log Comparisons ---")
for col in target_cols:
    tname = col.replace("target_", "")
    
    oth_vals = df_others[col].dropna().values
    z07_vals = df_z07[col].dropna().values
    z04_vals = df_z04[col].dropna().values
    
    oth_min, oth_max = np.min(oth_vals), np.max(oth_vals)
    z07_min, z07_max = np.min(z07_vals), np.max(z07_vals)
    z04_min, z04_max = np.min(z04_vals), np.max(z04_vals)
    
    # Check overlap: what percentage of Z-04's values are covered by Z-07 vs others?
    outside_oth = np.sum((z04_vals < oth_min) | (z04_vals > oth_max)) / len(z04_vals) * 100
    outside_z07 = np.sum((z04_vals < z07_min) | (z04_vals > z07_max)) / len(z04_vals) * 100
    
    # Combined training range (Others + Z-07)
    train_min, train_max = min(oth_min, z07_min), max(oth_max, z07_max)
    outside_train = np.sum((z04_vals < train_min) | (z04_vals > train_max)) / len(z04_vals) * 100
    
    print(f"\nTarget: {tname}")
    print(f"  4 Wells: [{oth_min:.3f}, {oth_max:.3f}] | Mean: {np.mean(oth_vals):.3f}")
    print(f"  Z-07:    [{z07_min:.3f}, {z07_max:.3f}] | Mean: {np.mean(z07_vals):.3f}")
    print(f"  Z-04:    [{z04_min:.3f}, {z04_max:.3f}] | Mean: {np.mean(z04_vals):.3f}")
    print(f"  % Z-04 outside 4 Wells: {outside_oth:.1f}% | outside Z-07: {outside_z07:.1f}% | outside combined Train: {outside_train:.1f}%")
    
    target_res.append({
        "Target": tname,
        "Others Min": oth_min,
        "Others Max": oth_max,
        "Others Mean": np.mean(oth_vals),
        "Z-07 Min": z07_min,
        "Z-07 Max": z07_max,
        "Z-07 Mean": np.mean(z07_vals),
        "Z-04 Min": z04_min,
        "Z-04 Max": z04_max,
        "Z-04 Mean": np.mean(z04_vals),
        "% Z-04 outside 4 Wells": outside_oth,
        "% Z-04 outside Z-07": outside_z07,
        "% Z-04 outside Train": outside_train
    })

df_t_res = pd.DataFrame(target_res)
df_t_res.to_csv(os.path.join(diagnostics_dir, "z07_vs_z04_targets_comparison.csv"), index=False)

# 2. Key out-of-range Seismic Attributes Comparison
key_attrs = ["attr_env_shift_-1", "attr_env_shift_-2", "attr_env_center", "attr_win_max", "attr_amp_shift_-2"]
attr_res = []

print("\n--- Key Seismic Attribute Comparisons ---")
for col in key_attrs:
    aname = col.replace("attr_", "")
    oth_vals = df_others[col].dropna().values
    z07_vals = df_z07[col].dropna().values
    z04_vals = df_z04[col].dropna().values
    
    oth_min, oth_max = np.min(oth_vals), np.max(oth_vals)
    z07_min, z07_max = np.min(z07_vals), np.max(z07_vals)
    z04_min, z04_max = np.min(z04_vals), np.max(z04_vals)
    
    outside_oth = np.sum((z04_vals < oth_min) | (z04_vals > oth_max)) / len(z04_vals) * 100
    outside_z07 = np.sum((z04_vals < z07_min) | (z04_vals > z07_max)) / len(z04_vals) * 100
    
    train_min, train_max = min(oth_min, z07_min), max(oth_max, z07_max)
    outside_train = np.sum((z04_vals < train_min) | (z04_vals > train_max)) / len(z04_vals) * 100
    
    print(f"\nAttribute: {aname}")
    print(f"  4 Wells: [{oth_min:.1f}, {oth_max:.1f}]")
    print(f"  Z-07:    [{z07_min:.1f}, {z07_max:.1f}]")
    print(f"  Z-04:    [{z04_min:.1f}, {z04_max:.1f}]")
    print(f"  % Z-04 outside 4 Wells: {outside_oth:.1f}% | outside Z-07: {outside_z07:.1f}% | outside combined Train: {outside_train:.1f}%")
    
    attr_res.append({
        "Attribute": aname,
        "Others Min": oth_min,
        "Others Max": oth_max,
        "Z-07 Min": z07_min,
        "Z-07 Max": z07_max,
        "Z-04 Min": z04_min,
        "Z-04 Max": z04_max,
        "% Z-04 outside 4 Wells": outside_oth,
        "% Z-04 outside Z-07": outside_z07,
        "% Z-04 outside Train": outside_train
    })
df_a_res = pd.DataFrame(attr_res)
df_a_res.to_csv(os.path.join(diagnostics_dir, "z07_vs_z04_attributes_comparison.csv"), index=False)

print("\nDiagnostics CSV files saved to diagnostics/ folder.")
