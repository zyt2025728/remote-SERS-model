# Level 1 — COMSOL-calibrated local hotspot layer

Level 1 loads a rectilinear table of COMSOL-derived local field-enhancement
factors and interpolates

```text
F_gap(gap_nm, particle_radius_nm, wavelength_nm, polarization_angle_deg)
```

inside the measured calibration domain. It applies the intended relationship

```text
E_hotspot = F_gap * E_drive
```

and a later electromagnetic SERS analysis may evaluate a relationship of the
form

```text
SERS enhancement ~ |E_hotspot|^4
```

That proportionality is documented here; it is not presented as physical
validation or as a complete SERS simulation.

## Calibration status: pending

No real COMSOL results are included. The file
`calibration/hotspot_lookup_template.csv` contains column headings only and is
deliberately rejected as calibration input until populated. Required real data
must cover a complete rectilinear grid of gap, radius, wavelength, and
polarization angle, with a non-negative field-enhancement value at every grid
point. The COMSOL model, materials, excitation normalization, mesh-convergence
evidence, and data provenance must accompany the populated table.

Synthetic values exist only inside unit-test temporary files. They are used to
verify interpolation and error handling and must never be reported as physical
simulation results.

## Future Level 3 connection

Level 3 will identify interparticle gaps in an aggregate and query this Level 1
interface using each gap's geometry and excitation conditions. Its complex
coupled-dipole drive field will provide `E_drive`; `F_gap` will then provide the
calibrated local gap correction. Level 3 is intentionally not implemented here.
