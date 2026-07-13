# Lasso-Selected Machine Learning Model Training Report
Report Generated: 2026-07-13 11:49:55

This report summarizes the performance of Random Forest and XGBoost models trained on features selected dynamically using LassoCV regularization.

## 1. Feature Engineering & Selection
- **Initial Attribute Pool**: 42 context attributes (center values, shifts, window statistics, relative position, acoustic impedance).
- **Feature Selector**: LassoCV (selects features with non-zero coefficients dynamically for each target).
- **Data Resolution**: Block-Averaged (matched to 2ms seismic bins).
- **Validation Strategy**: Leave-One-Well-Out Cross-Validation.

## 2. Model Performance Summary

Blind Well Evaluated: **Z-04**

| Target Property | Best Model | Best CV R2 | Best CV MAE | True Blind R2 | True Blind MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DT** | **XGBoost** | **-0.2431** | **3.9756** | **-4.7976** | 4.2336 |
| **GR** | **Random Forest** | **-0.2634** | **15.3153** | **-0.6567** | 20.1650 |
| **PHIE** | **XGBoost** | **-0.2957** | **0.0253** | **-1.0168** | 0.0259 |
| **PHIT** | **Random Forest** | **-0.5050** | **0.0211** | **-3.1081** | 0.0217 |
| **RHOB** | **XGBoost** | **-0.3826** | **0.0651** | **-0.5172** | 0.0977 |
| **SWE** | **Random Forest** | **-0.4724** | **0.2566** | **-0.5935** | 0.2250 |
| **VSH** | **XGBoost** | **-0.2924** | **0.0818** | **-1.1864** | 0.1085 |

## 3. Lasso-Selected Features per Target
- **DT**: `acoustic_impedance, env_ratio, env_shift_-2, ifreq_shift_+1, ifreq_shift_+2, ifreq_shift_-1, ifreq_shift_-2, rel_pos, win_min, win_std`
- **GR**: `ifreq_center, ifreq_shift_-2, ifreq_shift_-1, win_min, win_mean, win_max, rel_pos, win_std, ifreq_shift_+2, env_deriv`
- **PHIE**: `ifreq_center, ifreq_shift_-2, ifreq_shift_-1, win_std, win_mean, win_max, ifreq_shift_+2, win_min, env_deriv, rel_pos`
- **PHIT**: `ifreq_shift_+1, ifreq_shift_+2, ifreq_shift_-1, win_std, ifreq_shift_-2, win_max, win_mean, win_min, rel_pos, env_shift_-2`
- **RHOB**: `ifreq_center, ifreq_shift_+1, ifreq_shift_+2, ifreq_shift_-1, ifreq_shift_-2, rel_pos, win_max, win_mean, win_min, win_std`
- **SWE**: `ifreq_center, ifreq_shift_+1, ifreq_shift_+2, ifreq_shift_-1, ifreq_shift_-2, win_max, win_mean, win_std, win_min, rel_pos`
- **VSH**: `ifreq_center, ifreq_shift_-2, ifreq_shift_-1, win_mean, win_min, rel_pos, win_max, win_std, env_deriv, env_shift_+2`

## 4. Geophysical Interpretations
### Target: DT
- The model has a **POOR (Negative R2)** true blind validation score of **-4.7976** on Z-04. Although the wiggles might match the trend, the absolute values suffer from baseline shift. This is a common challenge for true blind well predictions where geological shifts occur.

### Target: GR
- The model has a **POOR (Negative R2)** true blind validation score of **-0.6567** on Z-04. Although the wiggles might match the trend, the absolute values suffer from baseline shift. This is a common challenge for true blind well predictions where geological shifts occur.

### Target: PHIE
- The model has a **POOR (Negative R2)** true blind validation score of **-1.0168** on Z-04. Although the wiggles might match the trend, the absolute values suffer from baseline shift. This is a common challenge for true blind well predictions where geological shifts occur.

### Target: PHIT
- The model has a **POOR (Negative R2)** true blind validation score of **-3.1081** on Z-04. Although the wiggles might match the trend, the absolute values suffer from baseline shift. This is a common challenge for true blind well predictions where geological shifts occur.

### Target: RHOB
- The model has a **POOR (Negative R2)** true blind validation score of **-0.5172** on Z-04. Although the wiggles might match the trend, the absolute values suffer from baseline shift. This is a common challenge for true blind well predictions where geological shifts occur.

### Target: SWE
- The model has a **POOR (Negative R2)** true blind validation score of **-0.5935** on Z-04. Although the wiggles might match the trend, the absolute values suffer from baseline shift. This is a common challenge for true blind well predictions where geological shifts occur.

### Target: VSH
- The model has a **POOR (Negative R2)** true blind validation score of **-1.1864** on Z-04. Although the wiggles might match the trend, the absolute values suffer from baseline shift. This is a common challenge for true blind well predictions where geological shifts occur.

