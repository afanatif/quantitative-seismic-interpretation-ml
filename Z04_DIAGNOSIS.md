# Diagnostics Report: Well Z-04 Coverage & Feasibility Analysis

## Executive Verdict (Plain Language for Geologists)

> [!IMPORTANT]
> **WELL Z-04 CONTAINS A SEVERE COVERAGE GAP.** The negative blind-test $R^2$ scores are primarily an expected consequence of data coverage gaps, rather than a modeling algorithm failure. The machine learning models are being asked to predict property values and handle seismic signatures that were **never shown in the training data**.

### Key Diagnostic Highlights:
- **Target Log Gaps**: Other petrophysical curves also fall out of training bounds, specifically: **GR** (3.8% of samples outside training range), **RHOB** (15.4% of samples outside training range), **VSH** (3.8% of samples outside training range).
- **Input Seismic Space Gaps**: Well Z-04 is located in a different seismic attribute domain. Out of 23 active features, **14** contain values completely outside the training ranges. The worst outliers are **env_shift_-1** (42.3% outside), **env_shift_-2** (38.5% outside), **env_center** (34.6% outside).
- **Spatial Isolation & Large Bulk Shift**:
  * Z-04 is the **northernmost well** in the survey and is geographically isolated, sitting **1013.9 meters** (about 3,326 ft) from its nearest neighbor (Z-02) and **3349.7 meters** on average from the southern training cluster.
  * Z-04 required a massive well-tie bulk shift of **+18.0 ms** to align with seismic reflectivity. In comparison, the 5 training wells have a mean bulk shift of only **-0.8 ms** (ranging from -10 to +6 ms). This massive shift suggests a major velocity anomaly or local stratigraphic thickness change that the model has no spatial baseline to understand.

---

## Detailed Diagnostic Evidence

### 1. Petrophysical Target Range Comparison (Block-Averaged ML Scale)

| Target | Train Min | Train Max | Z-04 Min | Z-04 Max | Train Mean | Z-04 Mean | % Z-04 Outside | below Min? | above Max? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GR** | 14.333 | 101.576 | 17.420 | 118.722 | 44.367 | 41.396 | 3.8% | no | YES |
| **DT** | 55.915 | 87.272 | 58.366 | 68.724 | 65.102 | 64.102 | 0.0% | no | no |
| **RHOB** | 2.286 | 2.689 | 2.154 | 2.575 | 2.541 | 2.452 | 15.4% | YES | no |
| **VSH** | 0.010 | 0.526 | 0.008 | 0.464 | 0.151 | 0.138 | 3.8% | YES | no |
| **PHIE** | 0.000 | 0.125 | 0.000 | 0.094 | 0.056 | 0.052 | 0.0% | no | no |
| **SWE** | 0.059 | 1.000 | 0.143 | 1.000 | 0.623 | 0.545 | 0.0% | no | no |
| **PHIT** | 0.024 | 0.136 | 0.053 | 0.103 | 0.076 | 0.074 | 0.0% | no | no |

### 2. Petrophysical Target Range Comparison (Raw Earth Log Scale)

| Target | Train Min | Train Max | Z-04 Min | Z-04 Max | Train Mean | Z-04 Mean | % Z-04 Outside | below Min? | above Max? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GR** | 9.185 | 188.438 | 9.360 | 176.077 | 44.367 | 40.325 | 0.0% | no | no |
| **DT** | 9.365 | 97.114 | 49.761 | 77.318 | 64.925 | 64.131 | 0.0% | no | no |
| **RHOB** | 1.720 | 2.831 | 1.855 | 2.750 | 2.541 | 2.434 | 0.0% | no | no |
| **VSH** | 0.000 | 0.949 | 0.000 | 0.660 | 0.150 | 0.135 | 0.0% | no | no |
| **PHIE** | 0.000 | 0.176 | 0.000 | 0.132 | 0.056 | 0.053 | 0.0% | no | no |
| **SWE** | 0.031 | 1.000 | 0.076 | 1.000 | 0.608 | 0.531 | 0.0% | no | no |
| **PHIT** | 0.000 | 0.188 | 0.024 | 0.139 | 0.076 | 0.074 | 0.0% | no | no |

### 3. Seismic Attribute Coverage Comparison

This table shows the features where Well Z-04 sits in a domain the model never saw. Ranked in descending order of out-of-range percentage:

| Attribute Feature | Train Min | Train Max | Z-04 Min | Z-04 Max | Train Std | Z-04 Mean | % Z-04 Outside | Max Out-of-Range Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `env_shift_-1` | 48.9341 | 19746.3014 | 1243.2925 | 22476.8463 | 5184.1312 | 14881.2332 | 42.3% | 3.06 σ |
| `env_shift_-2` | 38.4486 | 19406.9515 | 807.4690 | 22476.8463 | 4969.8937 | 14130.4752 | 38.5% | 3.27 σ |
| `env_center` | 48.9341 | 20774.3272 | 2334.6928 | 22476.8463 | 5387.6788 | 15584.5097 | 34.6% | 2.88 σ |
| `win_max` | -14052.5195 | 19241.4805 | -7956.3149 | 21966.3477 | 5919.4196 | 9007.3999 | 30.8% | 3.21 σ |
| `env_shift_+1` | 163.9448 | 21701.4455 | 3329.4155 | 22476.8463 | 5567.0572 | 16204.3917 | 23.1% | 2.72 σ |
| `amp_shift_-2` | -17640.3965 | 18615.7402 | -10896.6924 | 21966.3477 | 5672.9352 | 4024.7244 | 15.4% | 3.80 σ |
| `amp_shift_-1` | -17640.3965 | 19241.4805 | -10896.6924 | 21966.3477 | 6063.9984 | 3944.8952 | 15.4% | 3.55 σ |
| `win_mean` | -16354.7408 | 18021.2887 | -9720.5413 | 20375.0844 | 5875.6907 | 3537.7660 | 15.4% | 3.41 σ |
| `amp_center` | -17640.3965 | 19241.4805 | -10896.6924 | 21966.3477 | 6412.4514 | 3672.1452 | 15.4% | 3.37 σ |
| `amp_shift_+1` | -17640.3965 | 19241.4805 | -10896.6924 | 21966.3477 | 6718.2243 | 3259.6942 | 15.4% | 3.22 σ |
| `amp_shift_+2` | -20035.9941 | 19241.4805 | -11761.5088 | 21966.3477 | 6981.0352 | 2787.3711 | 15.4% | 3.11 σ |
| `env_deriv` | -670.3794 | 1689.1077 | -937.7750 | 1411.2037 | 455.9820 | 661.5793 | 11.5% | 2.86 σ |
| `win_min` | -20035.9941 | 16425.6543 | -11761.5088 | 17469.2988 | 5970.2224 | -2135.4324 | 7.7% | 3.31 σ |
| `env_shift_+2` | 163.9448 | 22274.8390 | 4550.6450 | 22476.8463 | 5723.8237 | 16743.6787 | 7.7% | 2.58 σ |
| `acoustic_impedance` | 0.0400 | 0.8000 | 0.1600 | 0.6400 | 0.1670 | 0.4108 | 0.0% | 0.00 σ |
| `env_ratio` | 1.0000 | 50.0000 | 1.0010 | 13.0579 | 10.5014 | 2.9659 | 0.0% | 0.00 σ |
| `ifreq_center` | -0.0000 | 116.9592 | 18.0683 | 47.1490 | 14.4227 | 23.5348 | 0.0% | 0.00 σ |
| `ifreq_shift_+1` | -0.0000 | 116.9592 | 17.4659 | 32.8487 | 13.8901 | 22.3932 | 0.0% | 0.00 σ |
| `ifreq_shift_+2` | -0.0000 | 116.9592 | 17.1187 | 28.4283 | 13.6931 | 21.7882 | 0.0% | 0.00 σ |
| `ifreq_shift_-1` | -125.0000 | 116.9592 | 18.0791 | 71.9904 | 18.3592 | 25.6088 | 0.0% | 0.00 σ |
| `ifreq_shift_-2` | -125.0000 | 177.4070 | 18.2910 | 87.1176 | 21.4673 | 28.2641 | 0.0% | 0.00 σ |
| `rel_pos` | 0.1090 | 0.4647 | 0.1186 | 0.1987 | 0.0905 | 0.1587 | 0.0% | 0.00 σ |
| `win_std` | 0.0000 | 7225.9523 | 791.8625 | 6974.8640 | 1608.4597 | 4031.5364 | 0.0% | 0.00 σ |

### 4. Spatial Well Network Analysis

Calculated 2D Euclidean distances from the blind holdout Z-04 to the training network:

| Well Name | distance (ft) | distance (m) | Well Inline | Well Crossline | Relative Geographic Position |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Z-02** | 3326.6 | 1013.9 | 535 | 193 | South of Z-04 |
| **Z-03** | 14598.3 | 4449.6 | 428 | 146 | South of Z-04 |
| **Z-05** | 16792.0 | 5118.2 | 398 | 199 | South of Z-04 |
| **Z-06** | 12241.4 | 3731.2 | 445 | 208 | South of Z-04 |
| **Z-07** | 7991.0 | 2435.7 | 488 | 199 | South of Z-04 |

### 5. Well-to-Seismic Tie Parameters Comparison

| Well Name | Inline | Crossline | Tie Correlation | Bulk Shift (ms) | Wavelet Freq (Hz) | Wavelet Phase (deg) | Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-02** | 535 | 193 | -0.8884 | -4.0 | 15.0 | 270.0 | GOOD |
| **Z-03** | 428 | 146 | 0.8948 | -10.0 | 20.0 | 270.0 | GOOD |
| **Z-04** | 569 | 193 | 0.9457 | 18.0 | 15.0 | 90.0 | GOOD |
| **Z-05** | 398 | 199 | 0.6680 | 6.0 | 15.0 | 90.0 | GOOD |
| **Z-06** | 445 | 208 | -0.7806 | 0.0 | 15.0 | 0.0 | GOOD |
| **Z-07** | 488 | 199 | 0.6985 | 4.0 | 20.0 | 90.0 | GOOD |

## Geophysical Recommendations

1. **Expand Training Well Coverage**: We must incorporate wells further north in the survey to capture the transition into this cleaner sandstone facies and calibrate the local seismic scaling. Well Z-08 (which is in the `las/` folder but currently unused) should be tie-tested to see if it fills this gap.
2. **Calibrate Predictions Using Histograms**: Standardize or transform predictions during inference. If Z-04's seismic attribute space is shifted, a simple local quantile transformation of prediction distributions to match historical training facies ranges can help mitigate tree bounding limitations.
3. **Velocity-Anomaly Investigation**: The 18ms shift is a structural warning sign. We should check if this well sits near a fault plane or is impacted by severe shallow gas/channel velocity anomalies that distort seismic frequencies and amplitudes.
