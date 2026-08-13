# Level 2 — stochastic aggregate and complex PDA

This directory contains one reproducible, representative realization generated
with 200 particles and random seed `20260813`. The aggregate is grown from a
connected random backbone, checked for overlap, and augmented by all near-contact
edges within the configured surface-gap cutoff. The realization contains dead
ends, branches, loops, variable coordination, and heterogeneous spatial density.

The electromagnetic calculation solves the full complex vector system

```text
p_i = alpha_i [E_inc(r_i) + sum_(j != i) G_ij p_j]
```

using the retarded 3D electric dyadic Green tensor for particle centers confined
to the xy plane. Excitation is a finite-width Gaussian centered on the leftmost
particle, not an illuminated boundary column.

## Scope and physical status

The representative run uses a 633 nm wavelength, vacuum host, spherical dipole
polarizability, and an explicitly recorded illustrative silver relative
permittivity assumption of `-15+1j`. A sourced wavelength-dependent silver
dielectric dataset is still required before physical validation. These outputs
demonstrate geometry generation and numerical solution only; they are not an
experimentally validated remote-SERS prediction.

No attenuation coefficient is fitted. Gap-hotspot networks and ensemble
statistics belong to later levels and are intentionally absent.

## Exact representative-run parameters

- Wavelength: 633 nm.
- Ag relative dielectric function: `-15+1j`, an **unsourced illustrative
  assumption** retained from the Level 2 run. It is not COMSOL data or a basis
  for physical validation; a sourced dispersive Ag dataset remains required.
- Surrounding refractive index: 1.0 (vacuum/air approximation).
- Sampled particle radius: mean 19.787664 nm; population standard deviation
  2.438066 nm.
- Near-contact surface gap: mean 2.571487 nm; median 2.523050 nm; population
  standard deviation 0.914231 nm; minimum 0.079093 nm; maximum 5.943767 nm.
- Gaussian waist parameter: 65 nm in the implemented field envelope
  `E = E0 exp(-r^2 / waist^2)`.
- Gaussian center: `(-424.737073, -464.237262)` nm, at particle 172.
- Polarization: `(1, 0, 0)` (x-linear).
- Peak incident-field amplitude: `1+0j` V/m.

“Directly illuminated” means that the particle-center incident intensity is at
least `1e-4` of its peak. The complementary set is the Level 2 remote region.
Six particles are directly illuminated and 194 are remote. Finite solved dipole
excitation occurs in that remote region; its strongest member is particle 62,
with normalized dipole intensity 0.00772936. This is a numerical observation for
one illustrative realization, not an attenuation fit or ensemble conclusion.

All electromagnetic matrix blocks use every pair `i != j`; the connectivity
graph does not truncate the complex Green-tensor coupling. `fields.csv` stores
the real and imaginary Cartesian components of `E_inc`, `E_scat`, and
`E_total = E_inc + E_scat` at every particle center.

The pre-existing Model 0 and 1D PDA sources were not present in this repository
checkout; Level 2 neither deletes nor replaces them.

## Files

- `particles.csv`: coordinates, radii, coordination, local incident magnitude,
  and solved dipole intensity.
- `edges.csv`: near-contact particle pairs and surface gaps.
- `summary.csv`: requested morphology, connectivity, source, and solver metrics.
- `aggregate_geometry.svg`: finite-radius particle geometry.
- `connectivity_graph.svg`: near-contact graph colored by coordination number.
- `dipole_intensity_map.svg`: solved complex-PDA dipole intensity and source center.
- Distribution CSV/SVG pairs report particle radius, surface gap, and
  coordination probability distributions.
- Three field-intensity SVGs report incident, scattered, and total fields.
- `normalized_dipole_intensity.svg` reports
  `log10(|p_i|^2 / max_j |p_j|^2)`.
- `geodesic_path.csv` and `.svg` record a natural connectivity path from the
  source particle to the strongest remote particle. Downstream-increase checks
  are reported without altering the realization.

Regenerate with:

```bash
python -m stochastic_2d.simulation --n-particles 200 --seed 20260813
```
