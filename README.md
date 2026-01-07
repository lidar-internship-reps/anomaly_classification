Adaptive Signal Processing Upgrade

Changes Made:
Introduced physiology-aware filtering by estimating dominant breathing frequency using PSD and dynamically adjusting the Butterworth low-pass cutoff. Added noise-aware smoothing that switches between Savitzky–Golay and B-spline methods based on signal quality to improve peak preservation and robustness.

True Inspiration–Expiration Analysis

Changes Made:
Replaced placeholder inspiration/expiration ratio with cycle-level computation using peak and valley detection. This enables accurate Ti/Te estimation, improving discrimination of abnormal breathing patterns such as apneustic and obstructive respiration.

Frequency-Domain Feature Expansion

Changes Made:
Added spectral features including dominant frequency, spectral entropy, and harmonic ratios derived from Welch PSD. These features enhance detection of irregular and periodic breathing disorders like Cheyne–Stokes and Biot’s respiration.

Chest–Abdomen Phase Coordination Analysis

Changes Made:
Implemented phase difference estimation between chest and abdomen signals using Hilbert transform. This allows explicit detection of asynchronous and paradoxical breathing patterns beyond simple correlation metrics.

Thoraco-Abdominal Contribution Index

Changes Made:
Introduced per-cycle amplitude contribution analysis to quantify relative chest versus abdomen dominance. Variability in contribution is tracked to identify respiratory fatigue and abnormal biomechanics.

Feature Fusion and Ablation Study

Changes Made:
Expanded the machine learning input from chest-only features to fused chest, abdomen, and synchrony features. Conducted ablation studies to quantify the contribution of each feature group to classification performance.

Temporal Respiratory Pattern Modeling

Changes Made:
Extended classification from static snapshots to temporal modeling using sliding windows and sequence learning (LSTM). This improves recognition of time-evolving respiratory disorders and long-cycle patterns.

Robustness and Noise Stress Testing

Changes Made:
Added synthetic noise, motion artifacts, and frame-drop simulations to evaluate system stability. Performance degradation trends are analyzed to assess real-world reliability under non-ideal conditions.

Confidence-Aware Disease Prediction

Changes Made:
Enhanced the classifier to output prediction confidence scores alongside disease labels. Low-confidence predictions are flagged to support clinical decision-making and reduce false certainty.

Automatic Chest and Abdomen ROI Tracking

Changes Made:
Replaced manual ROI selection with automatic depth-based tracking for chest and abdomen regions. The system dynamically reinitializes ROIs when tracking confidence decreases, improving usability.

Clinical Visualization and Dashboard

Changes Made:
Developed a real-time visualization dashboard displaying breathing signals, BPM trends, synchrony metrics, and disease predictions with confidence levels. Designed for clinician-friendly interpretation and continuous monitoring.
