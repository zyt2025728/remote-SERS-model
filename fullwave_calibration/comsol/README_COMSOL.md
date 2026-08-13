# COMSOL 3D isolated-dimer execution specification

**Status: generated but not executed/verified in COMSOL.** No COMSOL runtime or
Wave Optics license was detected in the present environment.

Use COMSOL Multiphysics with Wave Optics. The phasor convention is
`Re(E exp(-i omega t))`, so `curl(E)=i omega mu0 H`,
`curl(H)=-i omega epsilon0 epsilon_r E`, and the electric equation is
`curl(mu_r^-1 curl(E))-k0^2 epsilon_r E=0`. At Ag/medium interfaces enforce
continuous tangential E and H and, without imposed free charge, continuous
normal D. COMSOL's standard material continuity implements these conditions.

## Model

1. Create a 3D component in meters with parameters `R=10[nm]`, `g`,
   `lambda0=633[nm]`, `theta`, `E0=1[V/m]`, `nm=1`, and
   `epsAg=-15+i*1`. These exactly match the frozen Level-3A.5 illustrative
   local dielectric parameters; replace the Ag model only in a separately
   versioned study.
2. Create Ag spheres centered at `(-(R+g/2),0,0)` and `(R+g/2,0,0)`.
   The evaluation point is exactly `(0,0,0)`.
3. Surround them by a homogeneous spherical physical domain of radius at least
   `lambda0`; add a concentric PML of thickness at least `lambda0/2`.
4. Add **Electromagnetic Waves, Frequency Domain (ewfd)** in scattered-field
   formulation. Use a plane-wave background with complex amplitude
   `E0*(cos(theta),sin(theta),0)` and a propagation vector perpendicular to it.
5. Assign complex `epsAg` to both spheres and `nm^2` to background/PML. Use
   `mu_r=1` throughout.
6. Sweep gaps `{1,1.5,2,2.5,3,4,5,6}[nm]`, polarization angles
   `{0,30,60,90}[deg]`, and requested wavelengths without changing geometry
   conventions.

## Convergence and export

Use at least three meshes with gap maximum elements approximately `g/5`,
`g/8`, and `g/12`, boundary-layer/refined sphere surfaces, and adequate
wavelength resolution elsewhere. At the gap center export complex background,
scattered, and total `Ex,Ey,Ez`; `Eparallel=Ex`. Require consecutive magnitude
change below 2% and stable phase within 2 degrees. Repeat the fine mesh with a
second physical-domain/PML placement. Only rows passing both studies may be
marked `converged`. Export separate real/imaginary columns and all parameters to
CSV. Do not export only field magnitude.

The Java template beside this file establishes parameter, geometry, sweep, and
export intent. Physics/PML feature identifiers can vary by COMSOL release and
must be verified interactively before production execution.
