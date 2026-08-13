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

Regenerate with:

```bash
python -m stochastic_2d.simulation --n-particles 200 --seed 20260813
```
