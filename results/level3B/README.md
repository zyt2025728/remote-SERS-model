# Level 3B — full-wave calibration status

## A. Solver availability

No COMSOL executable/runtime was detected, so a Wave Optics license could not
be exercised. No independent validated full-wave Maxwell backend (BEM/FDTD/FEM)
is installed. Consequently **no full-wave or COMSOL field was fabricated** and
`solver_backend=PENDING_FULLWAVE`. The COMSOL package in
`fullwave_calibration/comsol/` is generated but not executed/verified.

## B. Maxwell formulation

The execution specification uses `exp(-i omega t)`,
`curl(E)=i omega mu0 H`, `curl(H)=-i omega epsilon0 epsilon_r E`, and
`curl(mu_r^-1 curl(E))-k0^2 epsilon_r E=0`, with complex Ag permittivity,
Maxwell interface continuity, a scattered-field formulation, and PML. Required
mesh and boundary convergence are detailed in `README_COMSOL.md`.

## C. Full-wave dimer results

None are usable yet. `USABLE_EGAP_TABLE.csv` and `gap_calibration_table.csv`
are deliberately header-only. There is no `Egap_COMSOL_dimer.csv` or mislabeled
fallback result. Production rows require an independent Maxwell solve and passed
mesh/PML convergence.

## D. PDA dimer results

`data/Egap_PDA_dimer.csv` contains the complex vector PDA reference for the
frozen parameters: R=10 nm, wavelength 633 nm, n=1, E0=1 V/m, and the existing
illustrative complex Ag permittivity `-15+1j`, over eight gaps and four angles.

## E–F. Correction and network application

No correction factor can be calculated without an independent full-wave
numerator. Therefore no network correction, ranking comparison, or
full-wave-calibrated electromagnetic SERS proxy is reported. Doing so would
manufacture calibration. The interpolation interface is implemented and rejects
empty, non-converged, non-independent, incomplete, or out-of-domain tables.

## G. Limitations

The remaining full-wave work requires COMSOL or another validated Maxwell
solver. The current local classical dielectric description lacks nonlocal and
quantum-tunneling corrections and is unsuitable below 1 nm. Even converged
classical full-wave values will still require experimental validation. No Level
4 ensemble work was performed.
