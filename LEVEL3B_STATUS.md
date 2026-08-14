# Level-3B completion status

The isolated MiePy calibration is complete: all 32 requested interacting
GMMT cases ran and passed two consecutive magnitude/phase checks. The actual
backend is `miepy_GMMT`; production data are in
`results/level3B/USABLE_EGAP_TABLE.csv`.

The repository supplied to this task contains **neither the stated
`results/level3B/data/Egap_PDA_dimer.csv` file nor a frozen Level-3A.5 hotspot
network**. It also contains no earlier stochastic-network implementation,
particle coordinates, PDA solver, hotspot table, or frozen random seed.
Consequently the requested PDA comparison and network correction cannot be
executed without inventing scientific inputs. No PDA, calibration-factor, or
network result has been fabricated.

`python -m fullwave_calibration.postprocess` implements the complete frozen-
input workflow and fails closed when either prerequisite is absent. The PDA
table must contain all 32 physical rows and complex `PDA_Ex_*`/`PDA_Ey_*`
columns. The network must contain its existing complex PDA field plus the
physical unit gap-axis columns `u_gap_x` and `u_gap_y`. Supplying those frozen
files will not rerun MiePy or regenerate geometry.

No COMSOL executable or license was detected. COMSOL numerical status is
**NOT EXECUTED**; no file named `Egap_COMSOL_dimer.csv` exists. The package is
provided for execution and validation on a licensed installation.
