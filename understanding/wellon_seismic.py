import os
import json
import numpy as np
import pandas as pd
import segyio
import lasio
import matplotlib.pyplot as plt

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
segy_path = os.path.abspath(os.path.join(script_dir, "..", "segy", "origional.segy"))
las_dir = os.path.abspath(os.path.join(script_dir, "..", "las"))
output_dir = os.path.abspath(os.path.join(script_dir, "..", "well_outputs", "well_seismic_tie"))

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Conversion factor
FEET_TO_METERS = 0.3048

# Well coordinates and Kelly Bushing (KB) elevations from the image
# (Note: Z-08 is excluded as requested)
well_data_feet = {
    "Z-02": {"x": 1205859.09, "y": 9692966.31, "kb": 146.46},
    "Z-03": {"x": 1201178.25, "y": 9682452.00, "kb": 147.64},
    "Z-04": {"x": 1205820.18, "y": 9696292.65, "kb": 147.64},
    "Z-05": {"x": 1206404.17, "y": 9679510.83, "kb": 144.36},
    "Z-06": {"x": 1207337.37, "y": 9684145.64, "kb": 146.98},
    "Z-07": {"x": 1206364.34, "y": 9688320.18, "kb": 147.97}
}

print("Step 1: Converting well coordinates to meters...")
well_data_meters = {}
for well_name, data in well_data_feet.items():
    well_data_meters[well_name] = {
        "x_feet": data["x"],
        "y_feet": data["y"],
        "kb_feet": data["kb"],
        "x_meters": round(data["x"] * FEET_TO_METERS, 2),
        "y_meters": round(data["y"] * FEET_TO_METERS, 2),
        "kb_meters": round(data["kb"] * FEET_TO_METERS, 2)
    }
    print(f"Well {well_name}:")
    print(f"  Feet:   X={data['x']:.2f}, Y={data['y']:.2f}")
    print(f"  Meters: X={well_data_meters[well_name]['x_meters']:.2f}, Y={well_data_meters[well_name]['y_meters']:.2f}")

print("\nStep 2: Loading seismic trace coordinates from SEG-Y...")
if not os.path.exists(segy_path):
    raise FileNotFoundError(f"SEG-Y file not found at {segy_path}")

with segyio.open(segy_path, "r", ignore_geometry=True) as f:
    # Read all trace coordinates and inline/crossline numbers
    src_x = f.attributes(segyio.TraceField.SourceX)[:].astype(float)
    src_y = f.attributes(segyio.TraceField.SourceY)[:].astype(float)
    inlines = f.attributes(segyio.TraceField.FieldRecord)[:]
    crosslines = f.attributes(segyio.TraceField.TraceNumber)[:]
    
    print(f"Seismic volume loaded with {f.tracecount} traces.")
    
    # 3D seismic survey outline
    segy_min_x, segy_max_x = src_x.min(), src_x.max()
    segy_min_y, segy_max_y = src_y.min(), src_y.max()

    mapping_results = {}
    
    print("\nStep 3: Mapping wells to the nearest trace and performing Log QC...")
    for well_name, well_m in well_data_meters.items():
        # Find nearest trace index in the SEG-Y file
        dx = src_x - well_m["x_meters"]
        dy = src_y - well_m["y_meters"]
        distances = np.sqrt(dx**2 + dy**2)
        nearest_idx = int(np.argmin(distances))
        offset = float(distances[nearest_idx])
        
        mapped_inline = int(inlines[nearest_idx])
        mapped_crossline = int(crosslines[nearest_idx])
        trace_x = float(src_x[nearest_idx])
        trace_y = float(src_y[nearest_idx])
        
        # Load well logs to check curves availability
        las_file_path = os.path.join(las_dir, f"{well_name}.las")
        available_curves = []
        has_required_logs = False
        missing_logs = []
        
        if os.path.exists(las_file_path):
            try:
                las = lasio.read(las_file_path)
                available_curves = list(las.keys())
                # Check for sonic (DT) and density (RHOB)
                required = ["DT", "RHOB"]
                missing_logs = [log for log in required if log not in available_curves]
                has_required_logs = len(missing_logs) == 0
            except Exception as e:
                print(f"  Error reading LAS for {well_name}: {e}")
        else:
            print(f"  Warning: LAS file not found for {well_name} at {las_file_path}")
            missing_logs = ["DT", "RHOB"]

        mapping_results[well_name] = {
            "well_coordinates_feet": {
                "x": well_m["x_feet"],
                "y": well_m["y_feet"],
                "kb": well_m["kb_feet"]
            },
            "well_coordinates_meters": {
                "x": well_m["x_meters"],
                "y": well_m["y_meters"],
                "kb": well_m["kb_meters"]
            },
            "mapped_trace": {
                "trace_index": nearest_idx,
                "inline": mapped_inline,
                "crossline": mapped_crossline,
                "trace_x_meters": trace_x,
                "trace_y_meters": trace_y,
                "offset_distance_meters": round(offset, 2)
            },
            "well_log_qc": {
                "las_file_found": os.path.exists(las_file_path),
                "available_curves": available_curves,
                "has_dt_and_rhob": has_required_logs,
                "missing_required_logs": missing_logs
            }
        }
        
        print(f"Well {well_name} mapped to:")
        print(f"  Trace Index: {nearest_idx}")
        print(f"  Inline:      {mapped_inline}")
        print(f"  Crossline:   {mapped_crossline}")
        print(f"  Offset Dist: {offset:.2f} meters")
        print(f"  Log QC:      {'PASS' if has_required_logs else 'FAIL (Missing ' + str(missing_logs) + ')'}")

# Write mapping results to JSON
json_output_path = os.path.join(output_dir, "well_to_trace_mapping.json")
with open(json_output_path, "w", encoding="utf-8") as f_json:
    json.dump(mapping_results, f_json, indent=2)
print(f"\nSaved mapping JSON to: {json_output_path}")

# Create visual QC plot
print("\nStep 4: Creating seismic-well QC map plot...")
plt.figure(figsize=(10, 8))

# Plot seismic grid outline or trace points
# Since 61,740 points might clutter the plot, we plot the bounds and a sample grid
# Plot every 5th trace to represent the grid
plt.scatter(src_x[::50], src_y[::50], c="lightgray", s=2, label="Seismic Traces (sampled)")

# Plot wells and highlight offsets
for well_name, res in mapping_results.items():
    wx = res["well_coordinates_meters"]["x"]
    wy = res["well_coordinates_meters"]["y"]
    tx = res["mapped_trace"]["trace_x_meters"]
    ty = res["mapped_trace"]["trace_y_meters"]
    
    # Plot well location (red triangle)
    plt.scatter(wx, wy, c="red", marker="^", s=100, edgecolors="black", zorder=5)
    # Plot closest trace (blue dot)
    plt.scatter(tx, ty, c="blue", marker="o", s=50, zorder=4)
    # Draw line connecting them (representing the offset)
    plt.plot([wx, tx], [wy, ty], "r--", alpha=0.5)
    
    # Label the well
    plt.text(wx + 100, wy + 100, f"{well_name}\n(Offset: {res['mapped_trace']['offset_distance_meters']}m)", 
             fontsize=9, weight="bold", bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

# Plot layout details
plt.title("Seismic Survey Grid & Well Mappings (QC Map)", fontsize=14, weight="bold")
plt.xlabel("X Coordinate (meters)", fontsize=12)
plt.ylabel("Y Coordinate (meters)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

# Add custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markeredgecolor='black', markersize=10, label='Well Location'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Closest Seismic Trace'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=4, label='Seismic Traces (Sampled)'),
    Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Well-to-Trace Offset')
]
plt.legend(handles=legend_elements, loc='upper left')

# Adjust limits with padding
padding = 1000
plt.xlim(segy_min_x - padding, segy_max_x + padding)
plt.ylim(segy_min_y - padding, segy_max_y + padding)

plot_output_path = os.path.join(output_dir, "seismic_well_map.png")
plt.tight_layout()
plt.savefig(plot_output_path, dpi=300)
plt.close()
print(f"Saved QC map plot to: {plot_output_path}")

print("\nAll steps completed successfully!")
