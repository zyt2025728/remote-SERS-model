# Level 2.6 physical-consistency validation

This is a separately generated validation run; it does not overwrite Level 2.
The primary realization has 200 monodisperse particles with `R=10 nm`, a
minimum permitted surface gap of 1 nm, and a connected stochastic topology.
Exact inputs and computed metrics are in `data/summary.csv`.

The PDA matrix contains all `N(N-1)=39,800` off-diagonal ordered pair
interactions. Graph edges are used only for topology and geodesic analysis. The
complex incident, scattered, and total Cartesian fields are retained in
`data/complex_fields.csv`; normalized maps clip only their displayed values,
while CSV values remain unmodified.

## Interpretation and limitations

**Direct illumination** is the field produced by the localized Gaussian source.
A particle is remote only when its normalized direct intensity is below `1e-6`
and its Euclidean distance from the Gaussian center exceeds 500 nm.

**Plasmon-mediated remote excitation** here means finite coupled-dipole/scattered
excitation in that remote set while the Gaussian field is negligible. This is a
single-realization numerical observation, not experimental validation.

**Network redistribution** refers to naturally non-monotonic dipole intensity
across edges directed by graph distance. Event tables include coordination,
local mean gaps, the connecting gap, and whether that edge participates in a
loop. These associations do not establish causality.

The Ag permittivity `-15+1j` remains an unsourced illustrative assumption. This
validation is not SERS, does not calculate an `|E|^4` enhancement, does not use
COMSOL hotspot calibration, fits no attenuation coefficient, and makes no
macroscopic or centimeter extrapolation.
