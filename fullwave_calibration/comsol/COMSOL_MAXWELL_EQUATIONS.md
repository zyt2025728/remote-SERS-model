# COMSOL 6.x EWFD model specification

Use a 3D **Wave Optics > Electromagnetic Waves, Frequency Domain (`ewfd`)**
study in scattered-field formulation with the `exp(-i omega t)` convention.
With `E=Eb+Esc`, EWFD solves
`curl(mu_r^-1 curl(E))-k0^2 epsilon_r E=0`; equivalently the scattered-field
equation has the residual of the prescribed background field on its right.
The evaluated gap field must be `ewfd.Ex+ewfd.relEx`, etc. (the exact variable
names must be confirmed against the installed COMSOL 6.x release).

The two 10 nm spheres are centered at `±(10[nm]+gap/2)` on x. Set Ag to
`epsilonr=-18.320291283372367+0.4792932084309133*i`, all other material to
air, wavelength to 633 nm, and background field to
`(cos(pol),sin(pol),0) V/m` propagating in +z. Surround an air sphere with a
spherical PML. Sweep `gap={1,1.5,2,2.5,3,4,5,6} nm` and
`pol={0,30,60,90} deg`.

Use quadratic curl elements, curvature refinement on both Ag surfaces, and a
gap region maximum element size sequence `{0.5,0.35,0.25,0.18} nm`. Repeat
with outer radii and PML thicknesses `(300,150)`, `(400,200)`, `(500,250) nm`.
Production requires two successive changes below 0.5% in magnitude and 0.5°
in phase. Export every trial to `COMSOL_mesh_convergence.csv` and
`COMSOL_PML_convergence.csv`; export only passing rows to
`Egap_COMSOL_dimer.csv` with the schema required by `import_comsol_results.py`.
