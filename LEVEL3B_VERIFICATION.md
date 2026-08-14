# Level-3B full-wave environment verification

The original isolated verification has been superseded by the complete
32-case adaptive calibration in `fullwave_calibration.calibration`. It uses
633 nm, MiePy's Johnson Ag data, air, 10 nm spheres, +z propagation, and
unit-amplitude polarization in the x-y plane. Every production row requires
two consecutive changes below 0.5% in magnitude and 0.5 degree in phase.

Run the original 3 nm longitudinal geometry with `python verify_level3b.py`,
or regenerate all production outputs with
`python -m fullwave_calibration.calibration`.
