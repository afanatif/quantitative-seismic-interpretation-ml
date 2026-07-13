import os
import segyio
import json
import numpy as np
import pandas as pd

# Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
segy_path = os.path.abspath(os.path.join(script_dir, "..", "segy", "origional.segy"))
output_dir = os.path.abspath(os.path.join(script_dir, "output"))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Starting SEG-Y extraction...")
print(f"Input SEG-Y file: {segy_path}")
print(f"Output directory: {output_dir}")

if not os.path.exists(segy_path):
    raise FileNotFoundError(f"SEG-Y file not found at {segy_path}")

with segyio.open(segy_path, "r", ignore_geometry=True) as f:
    # 1. Extract Text Header
    print("Extracting Text Header...")
    try:
        text_hdr = f.text[0]
        # segyio.tools.wrap wraps text at 80 characters per line
        decoded_text = segyio.tools.wrap(text_hdr)
        text_hdr_path = os.path.join(output_dir, "text_header.txt")
        with open(text_hdr_path, "w", encoding="utf-8") as th_file:
            th_file.write(decoded_text)
        print(f"Saved text header to {text_hdr_path}")
    except Exception as e:
        print(f"Error extracting text header: {e}")

    # 2. Extract Binary Header
    print("Extracting Binary Header...")
    try:
        bin_hdr = {}
        for key, val in f.bin.items():
            bin_hdr[str(key)] = val
        bin_hdr_path = os.path.join(output_dir, "binary_header.json")
        with open(bin_hdr_path, "w", encoding="utf-8") as bh_file:
            json.dump(bin_hdr, bh_file, indent=2)
        print(f"Saved binary header to {bin_hdr_path}")
    except Exception as e:
        print(f"Error extracting binary header: {e}")

    # 3. Extract Trace Headers
    print("Extracting Trace Headers (this might take a few seconds)...")
    try:
        # Standard populated trace header fields identified during inspection
        target_fields = [
            "TRACE_SEQUENCE_LINE",
            "TRACE_SEQUENCE_FILE",
            "FieldRecord",  # Inline
            "TraceNumber",   # Crossline
            "SourceX",
            "SourceY",
            "DelayRecordingTime",
            "TRACE_SAMPLE_COUNT",
            "TRACE_SAMPLE_INTERVAL"
        ]
        
        # Build headers list
        headers_list = []
        for i in range(f.tracecount):
            header_dict = {"TraceIndex": i}
            hdr = f.header[i]
            for field in target_fields:
                try:
                    # Convert field name to the segyio TraceField attribute
                    tf = getattr(segyio.TraceField, field)
                    header_dict[field] = hdr[tf]
                except AttributeError:
                    header_dict[field] = None
            headers_list.append(header_dict)
            
        df_headers = pd.DataFrame(headers_list)
        trace_hdrs_path = os.path.join(output_dir, "trace_headers.csv")
        df_headers.to_csv(trace_hdrs_path, index=False)
        print(f"Saved {len(df_headers)} trace headers to {trace_hdrs_path}")
    except Exception as e:
        print(f"Error extracting trace headers: {e}")

    # 4. Extract Seismic Trace Data (Amplitudes) & Sample Times
    print("Extracting Seismic Trace Data (Amplitudes)...")
    try:
        # Read the 2D seismic volume: tracecount x samplecount
        # segyio.tools.collect reads all traces into a 2D numpy array
        seismic_data = segyio.tools.collect(f.trace[:])
        seismic_data_path = os.path.join(output_dir, "seismic_data.npz")
        
        # Save both seismic data and sample times in a compressed npz file
        np.savez_compressed(
            seismic_data_path,
            amplitudes=seismic_data,
            samples=f.samples
        )
        print(f"Saved seismic data matrix {seismic_data.shape} to {seismic_data_path}")
    except Exception as e:
        print(f"Error extracting seismic data: {e}")

print("SEG-Y extraction complete successfully!")
