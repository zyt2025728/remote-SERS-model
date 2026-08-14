# Running the COMSOL package

COMSOL was not present in the cloud environment, so **COMSOL numerical status
= NOT EXECUTED** and no `Egap_COMSOL_dimer.csv` has been created.

On a licensed COMSOL Multiphysics 6.x installation with Wave Optics:

```bash
comsol compile build_dimer_model.java
comsol batch -inputfile build_dimer_model.class -outputfile dimer.log
python -m fullwave_calibration.import_comsol_results Egap_COMSOL_dimer.csv
```

Open the generated MPH file before the batch sweep and confirm the version-
specific total-field variable names. Run the mesh and PML convergence studies,
then the gap/polarization sweep. Never substitute MiePy rows for COMSOL output.
