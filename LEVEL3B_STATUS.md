# Level 3B final full-wave calibration

## Frozen inputs and provenance

No stochastic realization or MiePy case was rerun. The immutable realization is
assembled by hotspot ID from these integrated exports:

* `results/level2_6/data/particles.csv`: particle `id`, `x_nm`, `y_nm`,
  `radius_nm`, and `degree` (used only to construct each physical gap axis).
* `results/level3A/data/hotspots.csv`: `hotspot_id`, `particle_i`, `particle_j`,
  `x_nm`, `y_nm`, `gap_nm`, `euclidean_distance_nm`, complex
  `Etotal_[xyz]_[real|imag]`, and the original PDA metrics/classification.
* `results/level3A_5/data/corrected_hotspot_distances.csv`: `hotspot_id`,
  `corrected_geodesic_nm`, and `classification` (the authoritative corrected
  geodesic distance and remote-region classification).
* `results/level3A_5/data/downstream_enhancement_events.csv`: frozen hotspot
  neighbor pairs and upstream/downstream direction.

The matched dimer calls the validated `sphere_polarizability` (including its
radiation-reaction denominator), `electric_dyadic`, and
`solve_coupled_dipoles` implementations in `stochastic_2d/`. Network fields in
the frozen hotspot export were reconstructed by `fields_at_hotspots` in
`stochastic_2d/hotspots.py`.

## Validation gate

`fullwave_calibration.postprocess` defaults to a maximum 5% vector-magnitude
error and 2 degree phase error at every independent 30/60 degree case. The
observed magnitude errors (mean / median / maximum) are 0.0000221% /
0.0000146% / 0.0000973% at 30 degrees and 0.0135% / 0.000132% / 0.107% at 60
degrees. The gate passed before the frozen network correction was performed.

## Results and interpretation

The strongest PDA remote hotspot is 110 (particles 57–77, gap 2.811011 nm,
position 338.237140, 22.574023 nm, corrected geodesic distance 638.106456 nm),
with PDA M4 2.022256e-9 and corrected M4 1.161552e-8. The strongest corrected
remote hotspot is instead 146 (particles 82–97, gap 1.611473 nm, position
418.413580, -82.003291 nm, corrected geodesic distance 732.113204 nm), with
PDA M4 1.147594e-9 and corrected M4 2.696727e-8. Thus it moves to a different
physical gap. Their interpolated `(Cparallel, Cperp)` values are respectively
`(1.552280+0.003945i, 0.199960+0.005399i)` and
`(2.207657+0.010805i, 0.164581+0.003181i)`.

Remote rank Spearman correlation is 0.965726; top-5 overlap is 5/5 and top-10
overlap is 9/10. Hotspot 171 has the largest rank increase (13 to 7), while
hotspot 110 has the largest decrease (1 to 4). Maximum corrected source and
remote M4 are 1.184021e3 and 2.696727e-8, giving a corrected remote/source
ratio of 2.277601e-11, versus the Level-3A.5 PDA plasmon-remote/source value
1.600584e-11.

A. PDA substantially underestimates the longitudinal full-wave field most at
1–3 nm (`|Cparallel|` 3.19 down to 1.49), especially 1–2 nm. B. It trends
toward unity monotonically, reaching 1.151 at 6 nm, but has not reached unity.
C. The independently solved 30/60 degree fields pass the documented gate by a
large margin. D. The strongest remote hotspot moves from physical gap 110 to
146. E. Strong-event counts increase after correction for thresholds 2, 10,
and 100 (whole network: 130→160, 47→71, 11→20; remote: 10→15, 5→6, 1→3),
while the whole-network `Gamma_DE > 1` count changes 201→200 and remote changes
17→19. F. The high rank correlation shows network delivery remains dominant,
while the changed winner and threshold probabilities show that gap-scale
electrodynamics also matters: the result is controlled by both.

COMSOL numerical status = **NOT EXECUTED**. No COMSOL gap field was fabricated.
The current numerical full-wave result is `Egap_FW,dimer = Egap_MIEPY,dimer`,
with `solver_backend = miepy_GMMT`. These values are a full-wave-calibrated
electromagnetic SERS proxy, not experimental Raman intensity. Level 4 was not
performed.
