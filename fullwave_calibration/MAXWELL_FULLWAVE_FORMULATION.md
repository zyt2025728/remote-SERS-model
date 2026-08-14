# Frequency-domain full-wave formulation

We adopt phasors with physical fields `Re(E exp(-i omega t))`. Maxwell-Faraday
and source-free Maxwell-Ampère are

`curl E = i omega B` and `curl H = -i omega D`,

with `B = mu0 mu_r H` and `D = epsilon0 epsilon_r E`. Eliminating `H` gives

`curl(mu_r^-1 curl(E)) - k0^2 epsilon_r(r,omega) E = 0`, `k0=omega/c`.

At 633 nm the Johnson Ag data loaded by MiePy 1.1.0 are
`epsilon_Ag = -18.320291283372367 + 0.4792932084309133 i`; air has relative
permittivity 1 and relative permeability 1. All implementations must use
these same values.

With no imposed electric surface current or free surface charge, an interface
obeys continuity of `n x E`, `n x H`, and `n dot D`. The numerical field is
decomposed as `E_total = E_background + E_scattered`. Calibration samples the
**total complex field** at the physical gap center `(0,0,0)`, never only the
scattered field.
