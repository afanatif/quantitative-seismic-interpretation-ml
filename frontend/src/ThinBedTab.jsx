import React, { useState, useEffect, useRef } from 'react';
import { Layers, Sliders, Play, Info } from 'lucide-react';

export default function ThinBedTab({ wellsData }) {
  const [subTab, setSubTab] = useState('wedge'); // 'wedge' or 'logs'
  const [selectedWell, setSelectedWell] = useState('Z-02');
  const [frequency, setFrequency] = useState(30); // Ricker wavelet frequency (Hz)
  const wedgeCanvasRef = useRef(null);
  const tuningCanvasRef = useRef(null);

  // ── PART 1: Wedge Modeling Calculations & Rendering ──
  useEffect(() => {
    if (subTab !== 'wedge' || !wedgeCanvasRef.current || !tuningCanvasRef.current) return;

    const wCanvas = wedgeCanvasRef.current;
    const wCtx = wCanvas.getContext('2d');
    const tCanvas = tuningCanvasRef.current;
    const tCtx = tCanvas.getContext('2d');

    const W = wCanvas.width;
    const H = wCanvas.height;

    // Ricker Wavelet Generator (dt = 1ms)
    const dt = 1.0; // 1 ms
    const lengthMs = 80;
    const N_wave = Math.floor(lengthMs / dt);
    const half_wave = Math.floor(N_wave / 2);
    const wavelet = [];
    for (let i = 0; i < N_wave; i++) {
      const tSec = (i - half_wave) * (dt / 1000.0);
      const pft = Math.PI * frequency * tSec;
      wavelet.push((1 - 2 * pft * pft) * Math.exp(-pft * pft));
    }

    // Generate Wedge Traces
    // 60 traces across the screen: thickness goes from 0 to 45 ms TWT
    const nTraces = 60;
    const traceLength = 120; // 120 ms display window
    const topTime = 40; // Top sand at 40 ms
    const tracesData = [];
    const peakAmps = [];
    const thicknesses = [];

    for (let x = 0; x < nTraces; x++) {
      const thickness = x * (45.0 / (nTraces - 1)); // thickness in ms TWT
      thicknesses.push(thickness);

      // Reflectivity series: R_top = +1.0, R_bot = -1.0 (Soft Sand model)
      const reflectivity = new Array(traceLength).fill(0.0);
      reflectivity[topTime] = 1.0;
      const botTime = Math.min(traceLength - 1, topTime + Math.round(thickness));
      reflectivity[botTime] += -1.0; // sum if they overlap at thickness = 0

      // Convolve reflectivity with Ricker wavelet
      const convolved = new Array(traceLength).fill(0.0);
      for (let i = 0; i < traceLength; i++) {
        let val = 0.0;
        for (let w = 0; w < N_wave; w++) {
          const rIdx = i - (w - half_wave);
          if (rIdx >= 0 && rIdx < traceLength) {
            val += reflectivity[rIdx] * wavelet[w];
          }
        }
        convolved[i] = val;
      }
      tracesData.push(convolved);

      // Measure Peak Amplitude (Tuning curve)
      // Find the absolute maximum amplitude within the reservoir interval
      let maxVal = 0.0;
      for (let i = topTime - 10; i < Math.min(traceLength, botTime + 20); i++) {
        if (Math.abs(convolved[i]) > maxVal) {
          maxVal = Math.abs(convolved[i]);
        }
      }
      peakAmps.push(maxVal);
    }

    // Find Tuning Thickness (where peak amplitude is maximized)
    let maxAmpIdx = 0;
    let maxAmpVal = 0.0;
    for (let i = 2; i < peakAmps.length; i++) { // Skip near-zero thickness artifact
      if (peakAmps[i] > maxAmpVal) {
        maxAmpVal = peakAmps[i];
        maxAmpIdx = i;
      }
    }
    const tuningThicknessMs = thicknesses[maxAmpIdx];
    // Tuning thickness in meters: assuming sand velocity is 3000 m/s (1 ms TWT = 1.5 meters)
    const tuningThicknessM = (tuningThicknessMs * 1.5).toFixed(1);

    // ── DRAW WEDGE VARIABLE DENSITY CANVAS ──
    wCtx.fillStyle = '#0f172a';
    wCtx.fillRect(0, 0, W, H);

    const padLeft = 45;
    const padBottom = 30;
    const padTop = 15;
    const padRight = 15;
    const plotW = W - padLeft - padRight;
    const plotH = H - padTop - padBottom;
    const traceW = plotW / nTraces;

    // Paint Variable Density background
    for (let x = 0; x < nTraces; x++) {
      const trace = tracesData[x];
      const startX = padLeft + x * traceW;
      const endX = padLeft + (x + 1) * traceW;

      for (let y = 0; y < plotH; y++) {
        const sampleIdx = Math.floor((y / plotH) * traceLength);
        const val = trace[Math.min(sampleIdx, traceLength - 1)];

        // Red-White-Blue seismic scale
        let color = '';
        const norm = Math.max(-1.0, Math.min(1.0, val));
        if (norm > 0) {
          const r = 255;
          const g = Math.round(255 * (1 - norm));
          const b = Math.round(255 * (1 - norm));
          color = `rgb(${r}, ${g}, ${b})`;
        } else {
          const absNorm = Math.abs(norm);
          const r = Math.round(255 * (1 - absNorm));
          const g = Math.round(255 * (1 - absNorm));
          const b = 255;
          color = `rgb(${r}, ${g}, ${b})`;
        }

        wCtx.fillStyle = color;
        wCtx.fillRect(startX, padTop + y, Math.ceil(traceW), 1);
      }
    }

    // Overlay Wiggle curves (every 4 traces)
    wCtx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
    wCtx.lineWidth = 0.8;
    for (let x = 2; x < nTraces; x += 4) {
      const trace = tracesData[x];
      const centerX = padLeft + x * traceW + traceW / 2;

      wCtx.beginPath();
      for (let y = 0; y < plotH; y++) {
        const sampleIdx = Math.floor((y / plotH) * traceLength);
        const val = trace[Math.min(sampleIdx, traceLength - 1)];
        const dx = val * (traceW * 2.5); // scale wiggle deviation

        if (y === 0) wCtx.moveTo(centerX + dx, padTop);
        else wCtx.lineTo(centerX + dx, padTop + y);
      }
      wCtx.stroke();
    }

    // Draw Top Horizon (Flat)
    wCtx.strokeStyle = '#ffffff';
    wCtx.setLineDash([3, 3]);
    wCtx.lineWidth = 1.5;
    wCtx.beginPath();
    wCtx.moveTo(padLeft, padTop + (topTime / traceLength) * plotH);
    wCtx.lineTo(W - padRight, padTop + (topTime / traceLength) * plotH);
    wCtx.stroke();

    // Draw Bottom Horizon (Dipping)
    wCtx.beginPath();
    wCtx.moveTo(padLeft, padTop + (topTime / traceLength) * plotH);
    for (let x = 0; x < nTraces; x++) {
      const thickness = thicknesses[x];
      const botY = padTop + ((topTime + thickness) / traceLength) * plotH;
      const xPos = padLeft + x * traceW + traceW / 2;
      if (x === 0) wCtx.moveTo(xPos, botY);
      else wCtx.lineTo(xPos, botY);
    }
    wCtx.stroke();
    wCtx.setLineDash([]);

    // Draw Horizon text labels
    wCtx.fillStyle = '#ffffff';
    wCtx.font = 'italic 10px sans-serif';
    wCtx.fillText("Top Sand", padLeft + 10, padTop + (topTime / traceLength) * plotH - 5);
    wCtx.fillText("Base Sand (Dipping)", W - padRight - 110, padTop + ((topTime + thicknesses[nTraces - 1]) / traceLength) * plotH + 14);

    // Draw Axes Frame
    wCtx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    wCtx.lineWidth = 1;
    wCtx.strokeRect(padLeft, padTop, plotW, plotH);

    // X-Axis (Thickness in meters)
    wCtx.fillStyle = '#94a3b8';
    wCtx.font = '10px sans-serif';
    wCtx.textAlign = 'center';
    for (let x = 0; x <= nTraces; x += 15) {
      const frac = x / nTraces;
      const xPos = padLeft + frac * plotW;
      const meters = (frac * 45 * 1.5).toFixed(0); // 1 ms = 1.5m
      wCtx.beginPath();
      wCtx.moveTo(xPos, padTop + plotH);
      wCtx.lineTo(xPos, padTop + plotH + 4);
      wCtx.stroke();
      wCtx.fillText(`${meters}m`, xPos, padTop + plotH + 14);
    }
    wCtx.fillText("Wedge Layer Thickness (meters)", padLeft + plotW / 2, H - 4);

    // Y-Axis (Two-Way Time in ms)
    wCtx.textAlign = 'right';
    wCtx.textBaseline = 'middle';
    for (let t = 0; t <= traceLength; t += 30) {
      const yPos = padTop + (t / traceLength) * plotH;
      wCtx.beginPath();
      wCtx.moveTo(padLeft - 4, yPos);
      wCtx.lineTo(padLeft, yPos);
      wCtx.stroke();
      wCtx.fillText(`${t} ms`, padLeft - 8, yPos);
    }

    // ── DRAW TUNING CURVE CANVAS ──
    const tW = tCanvas.width;
    const tH = tCanvas.height;
    tCtx.fillStyle = '#0f172a';
    tCtx.fillRect(0, 0, tW, tH);

    const tPadL = 45;
    const tPadB = 30;
    const tPadT = 20;
    const tPadR = 15;
    const tPlotW = tW - tPadL - tPadR;
    const tPlotH = tH - tPadT - tPadB;

    // Draw Gridlines
    tCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    tCtx.lineWidth = 1;
    for (let val = 0.2; val <= 1.4; val += 0.2) {
      const yVal = tPadT + tPlotH * (1.0 - (val / 1.5));
      tCtx.beginPath();
      tCtx.moveTo(tPadL, yVal);
      tCtx.lineTo(tW - tPadR, yVal);
      tCtx.stroke();
    }

    // Plot peak amplitudes
    tCtx.strokeStyle = 'var(--accent-red)';
    tCtx.lineWidth = 2.0;
    tCtx.beginPath();
    for (let x = 0; x < nTraces; x++) {
      const thick = thicknesses[x] * 1.5; // meters
      const amp = peakAmps[x];
      // Map x to plot width (thickness max: 45*1.5 = 67.5m)
      const xPos = tPadL + (thick / 67.5) * tPlotW;
      const yPos = tPadT + tPlotH * (1.0 - (amp / 1.6)); // max amp around 1.6
      if (x === 0) tCtx.moveTo(xPos, yPos);
      else tCtx.lineTo(xPos, yPos);
    }
    tCtx.stroke();

    // Highlight tuning point
    const tuneX = tPadL + ((tuningThicknessMs * 1.5) / 67.5) * tPlotW;
    const tuneY = tPadT + tPlotH * (1.0 - (maxAmpVal / 1.6));
    tCtx.fillStyle = 'var(--accent-gold)';
    tCtx.beginPath();
    tCtx.arc(tuneX, tuneY, 5, 0, 2 * Math.PI);
    tCtx.fill();
    tCtx.strokeStyle = '#ffffff';
    tCtx.lineWidth = 1.5;
    tCtx.stroke();

    // Tuning point label
    tCtx.fillStyle = '#ffffff';
    tCtx.font = 'bold 9.5px sans-serif';
    tCtx.textAlign = 'left';
    tCtx.fillText(`Tuning Limit: ~${tuningThicknessM}m (${tuningThicknessMs.toFixed(1)} ms)`, tuneX + 8, tuneY - 4);

    // Axes lines
    tCtx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    tCtx.strokeRect(tPadL, tPadT, tPlotW, tPlotH);

    // X-Ticks
    tCtx.fillStyle = '#94a3b8';
    tCtx.font = '9px sans-serif';
    tCtx.textAlign = 'center';
    tCtx.textBaseline = 'top';
    for (let m = 0; m <= 60; m += 15) {
      const xPos = tPadL + (m / 67.5) * tPlotW;
      tCtx.beginPath();
      tCtx.moveTo(xPos, tPadT + tPlotH);
      tCtx.lineTo(xPos, tPadT + tPlotH + 4);
      tCtx.stroke();
      tCtx.fillText(`${m}m`, xPos, tPadT + tPlotH + 6);
    }
    tCtx.fillText("Thickness (meters)", tPadL + tPlotW / 2, tH - 4);

    // Y-Ticks
    tCtx.textAlign = 'right';
    tCtx.textBaseline = 'middle';
    for (let val = 0.0; val <= 1.5; val += 0.5) {
      const yPos = tPadT + tPlotH * (1.0 - (val / 1.6));
      tCtx.beginPath();
      tCtx.moveTo(tPadL - 4, yPos);
      tCtx.lineTo(tPadL, yPos);
      tCtx.stroke();
      tCtx.fillText(val.toFixed(1), tPadL - 8, yPos);
    }

    tCtx.save();
    tCtx.translate(14, tPadT + tPlotH / 2);
    tCtx.rotate(-Math.PI / 2);
    tCtx.textAlign = 'center';
    tCtx.fillText("Seismic Response Amplitude", 0, 0);
    tCtx.restore();

  }, [frequency, subTab]);

  // ── PART 2: Render Multi-Track Well Log Viewer ──
  const renderWellLogs = () => {
    const well = wellsData[selectedWell];
    if (!well) return null;

    // Filter samples that have actual logs (reservoir interval) to make the plot clean and high-resolution
    const reservoirSamples = well.samples.filter(s => s["time"] >= 2086.0 && s["time"] <= 2154.0);
    if (reservoirSamples.length === 0) {
      return (
        <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '40px' }}>
          No logged reservoir interval samples available for Well {selectedWell}.
        </div>
      );
    }

    // Get time bounds
    const times = reservoirSamples.map(s => s.time);
    const tMin = Math.min(...times);
    const tMax = Math.max(...times);
    const tRange = tMax - tMin || 1;

    // Dimensions of SVG tracks
    const trackW = 160;
    const trackH = 420;
    const padding = 10;
    const plotW = trackW - padding * 2;
    const plotH = trackH - padding * 3 - 20;

    // Normalize Y time to coordinate
    const getY = (timeVal) => padding + 20 + ((timeVal - tMin) / tRange) * plotH;

    // Helper to generate path for a property
    const getPath = (propName, minVal, maxVal, clampMin = null, clampMax = null) => {
      const pts = [];
      const range = maxVal - minVal || 1;
      
      reservoirSamples.forEach(s => {
        let val = s[propName];
        if (val === null || val === undefined) return;
        if (clampMin !== null) val = Math.max(clampMin, val);
        if (clampMax !== null) val = Math.min(clampMax, val);
        
        const norm = (val - minVal) / range;
        const x = padding + norm * plotW;
        const y = getY(s.time);
        pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
      });

      return pts.length > 0 ? `M ${pts.join(' L ')}` : '';
    };

    // Draw Y-axis ticks on the left of Track 1
    const renderYAxisTicks = () => {
      const ticks = [];
      const step = 10;
      const startT = Math.ceil(tMin / step) * step;
      for (let t = startT; t <= tMax; t += step) {
        ticks.push(
          <g key={t}>
            <line x1={0} y1={getY(t)} x2={6} y2={getY(t)} stroke="rgba(255,255,255,0.4)" strokeWidth={1} />
            <text x={-6} y={getY(t) + 3} fill="#94a3b8" fontSize={9.5} textAnchor="end">{t} ms</text>
            <line x1={8} y1={getY(t)} x2={trackW} y2={getY(t)} stroke="rgba(255,255,255,0.05)" strokeWidth={1} strokeDasharray="3 3" />
          </g>
        );
      }
      return ticks;
    };

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>Select Target Well:</span>
            <select
              value={selectedWell}
              onChange={(e) => setSelectedWell(e.target.value)}
              style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)', color: '#fff', fontSize: '13px' }}
            >
              {Object.keys(wellsData).map(wName => (
                <option key={wName} value={wName}>{wName} {wName === 'Z-04' ? '(Blind)' : ''}</option>
              ))}
            </select>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Logged Reservoir interval: <strong>{tMin.toFixed(0)} - {tMax.toFixed(0)} ms TWT</strong>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', justifyContent: 'center' }}>
          {/* TRACK 1: RAW SEISMIC AMPLITUDE */}
          <div className="glass-card" style={{ padding: '12px', width: `${trackW + 40}px` }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '12.5px', color: 'var(--text-primary)', textAlign: 'center' }}>Track 1: Raw Seismic</h4>
            <div style={{ display: 'flex', fontSize: '9px', justifyContent: 'space-between', color: 'var(--text-muted)', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px', marginBottom: '8px' }}>
              <span>-15k</span>
              <span>Amplitude</span>
              <span>+15k</span>
            </div>
            <svg width={trackW + 30} height={trackH} style={{ background: '#090d16', borderRadius: '4px' }}>
              <g transform="translate(30, 0)">
                {renderYAxisTicks()}
                <line x1={padding} y1={padding + 20} x2={padding} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                <line x1={padding + plotW} y1={padding + 20} x2={padding + plotW} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                <path d={getPath('seismic_amp', -15000, 15000)} fill="none" stroke="var(--accent-blue)" strokeWidth={1.5} />
              </g>
            </svg>
          </div>

          {/* TRACK 2: ENVELOPE & SWEETNESS */}
          <div className="glass-card" style={{ padding: '12px', width: `${trackW}px` }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '12.5px', color: 'var(--text-primary)', textAlign: 'center' }}>Track 2: Attributes</h4>
            <div style={{ display: 'flex', fontSize: '9px', justifyContent: 'space-between', color: 'var(--text-muted)', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px', marginBottom: '8px' }}>
              <span style={{ color: 'var(--accent-pink)' }}>Env</span>
              <span>Attributes</span>
              <span style={{ color: 'var(--accent-cyan)' }}>Sweetness</span>
            </div>
            <svg width={trackW} height={trackH} style={{ background: '#090d16', borderRadius: '4px' }}>
              <g>
                <line x1={padding} y1={padding + 20} x2={padding} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                <line x1={padding + plotW} y1={padding + 20} x2={padding + plotW} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                
                {/* Envelope path */}
                <path d={getPath('seismic_amp', 0, 18000)} fill="none" stroke="var(--accent-pink)" strokeWidth={1.2} />
                
                {/* Sweetness path: envelope / sqrt(ifreq) proxy */}
                {(() => {
                  const pts = [];
                  reservoirSamples.forEach(s => {
                    const env = s.seismic_amp ? Math.abs(s.seismic_amp) : 0;
                    // Mock instantaneous frequency or relative changes from trace
                    const sweetVal = env / 40.0; // scaled wrapper
                    const norm = Math.max(0.0, Math.min(1.0, sweetVal / 250.0));
                    const x = padding + norm * plotW;
                    const y = getY(s.time);
                    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
                  });
                  const pathD = pts.length > 0 ? `M ${pts.join(' L ')}` : '';
                  return <path d={pathD} fill="none" stroke="var(--accent-cyan)" strokeWidth={1.5} strokeDasharray="3 2" />;
                })()}
              </g>
            </svg>
          </div>

          {/* TRACK 3: ACOUSTIC IMPEDANCE */}
          <div className="glass-card" style={{ padding: '12px', width: `${trackW}px` }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '12.5px', color: 'var(--text-primary)', textAlign: 'center' }}>Track 3: Inversion</h4>
            <div style={{ display: 'flex', fontSize: '9px', justifyContent: 'space-between', color: 'var(--text-muted)', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px', marginBottom: '8px' }}>
              <span>Low AI</span>
              <span>Impedance</span>
              <span>High AI</span>
            </div>
            <svg width={trackW} height={trackH} style={{ background: '#090d16', borderRadius: '4px' }}>
              <g>
                <line x1={padding} y1={padding + 20} x2={padding} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                <line x1={padding + plotW} y1={padding + 20} x2={padding + plotW} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                
                {/* Acoustic Impedance predictions */}
                <path d={getPath('DT (Pred)', 100, 60)} fill="none" stroke="var(--accent-gold)" strokeWidth={1.5} />
              </g>
            </svg>
          </div>

          {/* TRACK 4: THIN-BED RESOLUTION (ACTUAL VS ML GR) */}
          <div className="glass-card" style={{ padding: '12px', width: `${trackW + 50}px` }}>
            <h4 style={{ margin: '0 0 6px 0', fontSize: '12.5px', color: 'var(--text-primary)', textAlign: 'center' }}>Track 4: Thin-Bed GR</h4>
            <div style={{ display: 'flex', fontSize: '9px', justifyContent: 'space-between', color: 'var(--text-muted)', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px', marginBottom: '8px' }}>
              <span>Sand</span>
              <span>Gamma Ray (GR)</span>
              <span>Shale</span>
            </div>
            <svg width={trackW + 50} height={trackH} style={{ background: '#090d16', borderRadius: '4px' }}>
              <g>
                <line x1={padding} y1={padding + 20} x2={padding} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                <line x1={padding + plotW} y1={padding + 20} x2={padding + plotW} y2={padding + 20 + plotH} stroke="rgba(255,255,255,0.2)" />
                
                {/* 1. Actual GR Log (Black line) */}
                <path d={getPath('GR (Act)', 30, 140)} fill="none" stroke="#fff" strokeWidth={1.8} />
                
                {/* 2. Predicted Raw GR (Smooth Blue line) */}
                <path d={getPath('GR (Pred Raw)', 30, 140)} fill="none" stroke="var(--accent-blue)" strokeWidth={1.2} strokeDasharray="3 3" />
                
                {/* 3. Predicted Calibrated GR (Upgraded Red line resolving thin beds) */}
                <path d={getPath('GR (Pred)', 30, 140)} fill="none" stroke="var(--accent-red)" strokeWidth={1.5} />
              </g>
            </svg>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '8px', fontSize: '9.5px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '3px', backgroundColor: '#fff' }}></span> Actual GR Well Log
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '3px', backgroundColor: 'var(--accent-blue)', strokeDasharray: '2 2' }}></span> Baseline ML (Raw Seismic)
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '3px', backgroundColor: 'var(--accent-red)' }}></span> Upgraded ML (Sweetness + Al)
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="slide">
      <div className="slide-title-area" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2>Sub-Seismic Thin Bed Resolution Workbench</h2>
          <p className="slide-subtitle">Analyze, simulate, and resolve thin sandstone reservoir beds that sit below traditional seismic tuning limits.</p>
        </div>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            onClick={() => setSubTab('wedge')}
            style={{
              padding: '6px 14px',
              borderRadius: '8px',
              border: `1.5px solid ${subTab === 'wedge' ? 'var(--accent-blue)' : 'var(--border-color)'}`,
              background: subTab === 'wedge' ? 'rgba(37,99,235,0.08)' : 'transparent',
              color: subTab === 'wedge' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              fontWeight: subTab === 'wedge' ? '600' : '400',
              fontSize: '12.5px',
              cursor: 'pointer'
            }}
          >
            1. Wedge Modeling Simulator
          </button>
          <button
            onClick={() => setSubTab('logs')}
            style={{
              padding: '6px 14px',
              borderRadius: '8px',
              border: `1.5px solid ${subTab === 'logs' ? 'var(--accent-blue)' : 'var(--border-color)'}`,
              background: subTab === 'logs' ? 'rgba(37,99,235,0.08)' : 'transparent',
              color: subTab === 'logs' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              fontWeight: subTab === 'logs' ? '600' : '400',
              fontSize: '12.5px',
              cursor: 'pointer'
            }}
          >
            2. Multi-Attribute Well QC
          </button>
        </div>
      </div>

      {subTab === 'wedge' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Left panel: Wedge canvas */}
          <div className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={18} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
                <h3 style={{ margin: 0 }}>Widess Wedge Seismic Model</h3>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                <span>Frequency: <strong>{frequency} Hz</strong></span>
                <input
                  type="range"
                  min="15"
                  max="60"
                  value={frequency}
                  onChange={(e) => setFrequency(parseInt(e.target.value))}
                  style={{ width: '100px', cursor: 'ew-resize' }}
                />
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', background: '#090d16', borderRadius: '6px', padding: '10px' }}>
              <canvas ref={wedgeCanvasRef} width={450} height={320} style={{ display: 'block', maxWidth: '100%' }} />
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontStyle: 'italic', display: 'flex', gap: '6px', alignItems: 'flex-start' }}>
              <Info size={14} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--accent-blue)' }} />
              <span>
                <strong>Widess wedge theory:</strong> Adjust the wavelet frequency slider. Higher frequencies compress the tuning window, allowing you to resolve thinner beds before peak interference occurs.
              </span>
            </div>
          </div>

          {/* Right panel: Tuning curve */}
          <div className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={18} className="card-icon" style={{ color: 'var(--accent-red)' }} />
              <h3 style={{ margin: 0 }}>Tuning Curve Analysis</h3>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', background: '#090d16', borderRadius: '6px', padding: '10px' }}>
              <canvas ref={tuningCanvasRef} width={450} height={320} style={{ display: 'block', maxWidth: '100%' }} />
            </div>
            <div style={{ fontSize: '11.5px', color: 'var(--text-primary)', background: 'var(--bg-dark)', padding: '10px', borderRadius: '6px', borderLeft: '3px solid var(--accent-gold)' }}>
              <strong>Seismic Tuning limit:</strong> Below the tuning thickness (gold dot), seismic amplitude decays rapidly to zero. ML models utilize sweetness/frequency phase changes to bypass this limit and reconstruct thin-bed formations.
            </div>
          </div>
        </div>
      )}

      {subTab === 'logs' && renderWellLogs()}
    </div>
  );
}
