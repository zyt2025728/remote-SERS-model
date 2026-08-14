# Level 4A 10-realization pilot summary

The network is a **2D stochastic retarded coupled-dipole approximation (CDA) model** with frozen full-wave MiePy/GMMT isolated-gap calibration. Corrected M4 is a full-wave-calibrated electromagnetic SERS proxy, not experimental Raman intensity.

1. **Exact seeds:** 20260813, 20365542, 20470271, 20575000, 20679729, 20784458, 20889187, 20993916, 21098645, 21203374.
2. **Frozen configuration:** see `FROZEN_CONFIGURATION.json`; only `random_seed` varies.
3. **Successful realizations:** 10.
4. **Failed realizations:** 0; no failure reasons.
5. **Calibration coverage:** all-hotspot range 1.000-1.000; remote range 1.000-1.000.
6. **Remote maximum M4 FWcorr:** median 1.22017e-08; range 3.28371e-09-7.28547e-08.
7. **eta_RS FWcorr:** median 5.84662e-12; range 6.0977e-14-2.69581e-11.
8. **Downstream probabilities (FWcorr):**
   - remote, P_Gamma_DE_gt_1: mean 0.4163, median 0.4227, range 0.3333-0.4863.
   - remote, P_Gamma_DE_ge_2: mean 0.3250, median 0.3225, range 0.2222-0.4444.
   - remote, P_Gamma_DE_ge_10: mean 0.1172, median 0.1496, range 0.0000-0.2022.
   - remote, P_Gamma_DE_ge_100: mean 0.0350, median 0.0522, range 0.0000-0.0682.
   - whole_network, P_Gamma_DE_gt_1: mean 0.4651, median 0.4626, range 0.4232-0.5100.
   - whole_network, P_Gamma_DE_ge_2: mean 0.3650, median 0.3710, range 0.3229-0.3969.
   - whole_network, P_Gamma_DE_ge_10: mean 0.1855, median 0.1852, range 0.1517-0.2179.
   - whole_network, P_Gamma_DE_ge_100: mean 0.0631, median 0.0596, range 0.0304-0.0933.
9. **Strongest remote identity changes:** 6/10 realizations.
10. **Remote rank correlation:** mean 0.9144, median 0.9261, range 0.8310-0.9657.
11. **Geodesic envelope:** fixed 100 nm bins show quantile propagation structure; no single exponential attenuation coefficient was fitted.
12. **Correlations:** pooled, realization-balanced, and strongest-per-realization exploratory Spearman results are recorded in `data/remote_hotspot_correlations.csv`; they are associations, not causal evidence.
13. **Pilot size:** 10 realizations are insufficient for a final ensemble claim; convergence code supports later N=10,20,30,40,50,75,100.
14. **Calibration coverage adequate:** yes for this pilot; outside-domain CDA values are retained without fabricated corrected values.
15. **Numerical instability:** none detected from residual and finite-value diagnostics.
16. **Recommendation: `PROCEED_TO_50`.**

No 50-realization run and no Level 4B work were performed.
