# Level-3B completion status

The isolated MiePy calibration is complete: all 32 requested interacting
GMMT cases ran and passed two consecutive magnitude/phase checks. The actual
backend is `miepy_GMMT`; production data are in
`results/level3B/USABLE_EGAP_TABLE.csv`.

The repository supplied to this task contains **no Level-2.6, Level-3A, or
Level-3A.5 stochastic-network implementation, data, particle coordinates, PDA
solver, hotspot table, or frozen random seed**. Consequently the requested
"exact same PDA implementation" cannot be executed, and Parts F–M cannot be
performed without inventing or redesigning scientific inputs. No PDA,
calibration-factor, or network result has been fabricated. Supply the frozen
network and its PDA implementation to complete those parts.

No COMSOL executable or license was detected. COMSOL numerical status is
**NOT EXECUTED**; no file named `Egap_COMSOL_dimer.csv` exists. The package is
provided for execution and validation on a licensed installation.
