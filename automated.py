"""
================================================================================
  AUTOMATED MASTER PIPELINE EXECUTION ENGINE (automated.py)
================================================================================
  Author: DeepMind Antigravity AI Engine
  Purpose:
    Executes the complete Quantitative Seismic Interpretation & ML Reservoir
    Prediction pipeline sequentially with strict error handling, step verification,
    and UTF-8 environment enforcement.

  Execution Order:
    Step 1: well_seismic/impedance_tie.py          (Well-Seismic Tie Physics & Synthetic Seismogram)
    Step 2: well_seismic/auto_well_seismic_aligner.py (Dynamic 3D Trace Energy Horizon Scanner)
    Step 3: ml_training/compute_thin_bed_attributes.py (CWT + SSWT Spectral Attribute Calculation)
    Step 4: ml_training/train_model_v11.py         (2-Stage ML Engine Training under LOGO-CV)
    Step 5: ml_training/generate_frontend_data.py  (Export Unified data.js Database)
    Step 6: ml_training/precompute_v11_slice_predictions.py (Precompute 2D/3D Slice Overlay Arrays)
    Step 7: ml_training/generate_grid_predictions.py (Precompute 3D Spatial Grid Arrays)
    Step 8: scratch/sanity_check.py                (End-to-End System Sanity Verification)

  Usage:
    python automated.py
================================================================================
"""

import os
import sys
import time
import subprocess

# ── Setup Environment & Paths ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Enforce UTF-8 encoding for subprocesses
EXEC_ENV = os.environ.copy()
EXEC_ENV["PYTHONIOENCODING"] = "utf-8"
EXEC_ENV["PYTHONUNBUFFERED"] = "1"

# ── Pipeline Execution Plan ───────────────────────────────────────────────────
STEPS = [
    {
        "step_num": 0,
        "name": "Verify & Install Python Dependencies (requirements.txt)",
        "script": None,
        "expected_output": os.path.join(PROJECT_ROOT, "requirements.txt"),
        "desc": "Verifies that all required Python packages (numpy, scipy, pandas, xgboost, lightgbm, lasio, segyio) are installed."
    },
    {
        "step_num": 1,
        "name": "Well-Seismic Tie Physics & Synthetic Seismogram Engine",
        "script": os.path.join("well_seismic", "impedance_tie.py"),
        "expected_output": os.path.join("well_seismic", "output", "impedance_tie_summary.csv"),
        "desc": "Calculates Acoustic Impedance (AI), Ricker wavelets, phase rotation (theta), and bulk time shifts against SEG-Y traces."
    },
    {
        "step_num": 2,
        "name": "Dynamic 3D Trace Energy Horizon Bounds Scanner",
        "script": os.path.join("well_seismic", "auto_well_seismic_aligner.py"),
        "expected_output": os.path.join("well_seismic", "output", "auto_well_horizon_mapping.json"),
        "desc": "Scans 3D volume trace at each borehole location to extract non-zero reservoir channel top (kFirst) and base (kLast)."
    },
    {
        "step_num": 3,
        "name": "CWT + SSWT Thin-Bed Spectral Attribute Computation",
        "script": os.path.join("ml_training", "compute_thin_bed_attributes.py"),
        "expected_output": os.path.join("frontend", "src", "thin_bed_data.js"),
        "desc": "Extracts 5 Morlet CWT envelopes and SSWT phase-reassigned frequency ratios for sub-seismic 3.2m bed resolution."
    },
    {
        "step_num": 4,
        "name": "Train Sand/Shale Facies Classifier Model",
        "script": os.path.join("ml_training", "train_facies_model.py"),
        "expected_output": os.path.join("ml_outputs_v11", "facies_model", "facies_model.joblib"),
        "desc": "Trains Random Forest classifier to predict Sand vs Shale probability (P_sand) for Facies Modulation."
    },
    {
        "step_num": 5,
        "name": "2-Stage Machine Learning Engine Training (LOGO-CV)",
        "script": os.path.join("ml_training", "train_model_v11.py"),
        "expected_output": os.path.join("ml_outputs_v11", "models"),
        "desc": "Trains Stage 1 Stacking Regressors, Stage 2 Cascaded XGBoost/LightGBM, and Sand-Probability Facies Modulation Engine."
    },
    {
        "step_num": 6,
        "name": "Regenerate Unified Frontend Database (data.js)",
        "script": os.path.join("ml_training", "generate_frontend_data.py"),
        "expected_output": os.path.join("frontend", "src", "data.js"),
        "desc": "Merges log samples, 3D seismic trace slices, V11 model predictions, and tie metadata into frontend/src/data.js."
    },
    {
        "step_num": 7,
        "name": "Precompute 2D Crossline Slice Overlay Arrays",
        "script": os.path.join("ml_training", "precompute_v11_slice_predictions.py"),
        "expected_output": os.path.join("frontend", "public", "v11_pred_dt.bin"),
        "desc": "Pre-calculates 2D vertical crossline property arrays for 60 FPS real-time UI rendering in React Canvas."
    },
    {
        "step_num": 8,
        "name": "Precompute 3D Spatial Grid Prediction Arrays",
        "script": os.path.join("ml_training", "generate_grid_predictions.py"),
        "expected_output": os.path.join("frontend", "public", "v11_pred_ai.bin"),
        "desc": "Pre-calculates 3D spatial property grids across the reservoir volume."
    },
    {
        "step_num": 9,
        "name": "End-to-End System Sanity Verification",
        "script": os.path.join("scratch", "sanity_check.py"),
        "expected_output": None,
        "desc": "Verifies 3D volume dimensions, physical target ranges, well alignments, and production build."
    }
]

def run_pipeline():
    total_start_time = time.time()
    force_rerun = "--force" in sys.argv
    
    print("=" * 80)
    print(" 🚀 AUTOMATED MASTER PIPELINE EXECUTION ENGINE (V11)")
    print("=" * 80)
    print(f" Working Directory: {PROJECT_ROOT}")
    print(f" Total Steps to Execute: {len(STEPS)}")
    print(f" Force Re-run Mode: {'ENABLED (--force)' if force_rerun else 'DISABLED (Auto-Skipping Verified Steps)'}")
    print("=" * 80 + "\n")
    
    for step in STEPS:
        step_num = step["step_num"]
        name     = step["name"]
        script   = step["script"]
        exp_out  = step["expected_output"]
        desc     = step["desc"]
        
        print(f"► [STEP {step_num}/{len(STEPS)-1}] {name}")
        print(f"  Info:   {desc}")
        
        # Check if output already exists (Skip unless --force is set)
        if exp_out and not force_rerun and script is not None:
            exp_out_path = os.path.join(PROJECT_ROOT, exp_out)
            if os.path.exists(exp_out_path):
                print(f"  ⏭️ [SKIPPED] Output already verified: {exp_out} (Use --force to re-run)")
                print("─" * 80)
                continue

        step_start_time = time.time()
        
        # Step 0 Special Case: Install/verify requirements.txt
        if script is None:
            req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
            if not os.path.exists(req_path):
                print(f"\n❌ STRICT ERROR: requirements.txt not found: {req_path}")
                sys.exit(1)
            cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
            print(f"  Executing: {' '.join(cmd)}")
            process = subprocess.run(cmd, cwd=PROJECT_ROOT, env=EXEC_ENV)
        else:
            script_path = os.path.join(PROJECT_ROOT, script)
            print(f"  Script: {script}")
            if not os.path.exists(script_path):
                print(f"\n❌ STRICT ERROR: Script file not found: {script_path}")
                sys.exit(1)
            cmd = [sys.executable, script_path]
            process = subprocess.run(cmd, cwd=PROJECT_ROOT, env=EXEC_ENV)
        
        step_duration = time.time() - step_start_time
        
        if process.returncode != 0:
            print("\n" + "!" * 80)
            print(f" ❌ STRICT PIPELINE FAILURE AT STEP {step_num}: {name}")
            print(f"    Failed Command: {' '.join(cmd)}")
            print(f"    Exit Code: {process.returncode}")
            print("!" * 80)
            sys.exit(process.returncode)
            
        # Verify expected output file if specified
        if exp_out:
            exp_out_path = os.path.join(PROJECT_ROOT, exp_out)
            if not os.path.exists(exp_out_path):
                print(f"\n❌ STRICT OUTPUT VERIFICATION FAILURE AT STEP {step_num}:")
                print(f"   Expected output not found: {exp_out_path}")
                sys.exit(1)
            print(f"  ✓ Output Verified: {exp_out}")
            
        print(f"  ✓ STEP {step_num} PASSED in {step_duration:.1f}s\n" + "─" * 80)
        
    total_duration = time.time() - total_start_time
    
    print("\n" + "=" * 80)
    print(" 🎉 SUCCESS! ALL 8 PIPELINE STEPS COMPLETED IN {:.1f}s!".format(total_duration))
    print("    Full 3D Seismic ML Pipeline is 100% Calibrated & Verified.")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
