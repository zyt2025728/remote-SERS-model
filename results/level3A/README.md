# Level 3A — uncalibrated gap-hotspot network

This run preserves the Level 2.6 geometry and full complex, all-particle PDA
solution. Each validated 1–6 nm near-contact gap becomes a hotspot at the
midpoint between its two closest particle-surface points. Every solved particle
dipole contributes to the complex scattered field at every hotspot. The hotspot
graph (shared-particle adjacency) is used only for topology and distance.

`M2=|E_total|^2/|E0|^2` and `M4=M2^2` are uncalibrated PDA-driven hotspot
metrics. `M4` is only an uncalibrated `|E|^4` hotspot proxy: it is not a
quantitatively accurate SERS enhancement factor. CSVs retain actual values;
plot floors affect visualization only.

## Interpretation

### A. Numerically demonstrated

The realization has remote gap excitation with negligible direct Gaussian
illumination, nonzero scattered fields, heterogeneous hotspot strengths, and
naturally increasing downstream hotspot transitions. Particle and gap rankings
are saved separately and are not assumed equivalent.

### B. Cautious physical inference

These numerical patterns are consistent with collective complex coupling,
multi-path redistribution, topology-dependent localization, and interference.
One realization cannot establish causality or population-level behavior.

### C. Not established

Level 3A provides no calibrated SERS enhancement, higher-order multipolar gap
correction, quantum-tunneling correction, validated absolute Raman intensity,
ensemble statistic, macroscopic attenuation coefficient, or centimeter-scale
extrapolation. COMSOL calibration is intentionally not applied.
