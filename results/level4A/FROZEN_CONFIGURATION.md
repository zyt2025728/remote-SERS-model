# Frozen Level 4A configuration

The network is a **2D stochastic retarded coupled-dipole approximation (CDA) model** with full-wave MiePy/GMMT isolated-gap calibration.

The **only** realization-dependent parameter is `random_seed`. All values below are identical for every pilot realization.

## Geometry and stochastic generator

- `n_particles`: `200`
- `radius_mean_nm`: `10.0`
- `radius_std_nm`: `0.0`
- `radius_min_nm`: `10.0`
- `radius_max_nm`: `10.0`
- `gap_mean_nm`: `3.0`
- `gap_std_nm`: `1.0`
- `gap_min_nm`: `1.0`
- `gap_max_nm`: `6.0`
- `positional_disorder_rad`: `0.45`
- `connection_gap_nm`: `6.0`
- `max_attempts_per_particle`: `4000`

## Electromagnetic, source, region, and analysis definitions

- **wavelength nm:** `633.0`
- **nanoparticle material:** `Ag sphere`
- **particle relative permittivity:** `{"real": -15.0, "imag": 1.0}`
- **surrounding medium:** `homogeneous vacuum, relative permittivity 1+0j`
- **source particle rule:** `particle with minimum x coordinate`
- **aggregate generation:** `validated connected stochastic growth in stochastic_2d.geometry.generate_aggregate`
- **branching parameters:** `four of every five children favor low-degree parents; every fifth favors compact centroid growth; angular disorder from geometry.positional_disorder_rad`
- **coordination constraints:** `no hard coordination cap; overlap/minimum-gap rejection and inverse-(1+degree) branch weighting`
- **minimum separation rule:** `all particle pairs have surface separation >= geometry.gap_min_nm`
- **particle radius distribution:** `frozen truncated normal specified by geometry radius fields`
- **surface gap distribution:** `frozen truncated normal specified by geometry gap fields`
- **gaussian beam waist nm:** `55.0`
- **incident field amplitude V per m:** `{"real": 1.0, "imag": 0.0}`
- **incident polarization xyz:** `[1.0, 0.0, 0.0]`
- **incident propagation convention:** `localized scalar Gaussian envelope; exp(-i omega t)`
- **hotspot definition:** `midpoint between facing surfaces of every 1-6 nm near contact`
- **near contact graph criterion nm:** `[1.0, 6.0]`
- **source region definition:** `M2_inc >= 1e-6`
- **transition region definition:** `not source and not remote`
- **remote region definition:** `M2_inc < 1e-6 and Euclidean source distance > 500 nm`
- **plasmon remote definition:** `remote and M2_scat/M2_inc >= 100`
- **geodesic distance definition:** `shortest hotspot-adjacency path weighted by midpoint distance`
- **geodesic launch definition:** `strongest CDA-M4 source hotspot`
- **downstream direction definition:** `strictly increasing corrected weighted geodesic distance`
- **CDA polarizability:** `Clausius-Mossotti sphere polarizability`
- **radiation correction:** `alpha=alpha_static/(1-i*k^3*alpha_static/(6*pi*epsilon0*epsilon_m))`
- **retarded dyadic Green tensor:** `full 3D electric dyadic retaining r^-3, r^-2, r^-1 terms`
- **CDA solver:** `dense complex self-consistent solve using all ordered particle pairs`
- **fullwave calibration table:** `results/level3B/data/gap_calibration_table.csv`
- **C parallel:** `piecewise-linear interpolation of complex real/imaginary columns; no extrapolation`
- **C perp:** `piecewise-linear interpolation of complex real/imaginary columns; no extrapolation`
- **calibration domain nm:** `[1.0, 6.0]`
- **solver backend:** `miepy_GMMT`
- **COMSOL numerical status:** `NOT EXECUTED`
- **geodesic bin edges nm:** `[0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0]`

Corrected M4 is a **full-wave-calibrated electromagnetic SERS proxy**, not experimental Raman intensity.
