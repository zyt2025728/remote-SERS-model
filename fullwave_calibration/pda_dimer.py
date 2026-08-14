"""Isolated dimer calculation using the frozen network PDA implementation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from stochastic_2d.coupled_dipole import solve_coupled_dipoles
from stochastic_2d.green_tensor import electric_dyadic
from stochastic_2d.polarizability import sphere_polarizability

@dataclass(frozen=True)
class DimerResult:
    gap_nm: float
    radius_nm: float
    wavelength_nm: float
    polarization_deg: float
    centers_m: np.ndarray
    gap_position_m: np.ndarray
    incident: np.ndarray
    scattered: np.ndarray
    total: np.ndarray
    dipoles: np.ndarray

def solve_pda_dimer(gap_nm: float, radius_nm: float = 10., wavelength_nm: float = 633.,
                    polarization_deg: float = 0., silver_permittivity: complex =
                    -18.320291283372367 + 0.4792932084309133j,
                    medium_permittivity: complex = 1+0j, amplitude: complex = 1+0j) -> DimerResult:
    """Solve the exact Level-3A.5 PDA for two spheres and sample the gap center."""
    if gap_nm < 1 or radius_nm <= 0 or wavelength_nm <= 0:
        raise ValueError("classical calibration requires gap >= 1 nm and positive radius/wavelength")
    half=(radius_nm+gap_nm/2)*1e-9
    centers=np.array([[-half,0.,0.],[half,0.,0.]])
    angle=np.deg2rad(polarization_deg);polarization=np.array([np.cos(angle),np.sin(angle),0.])
    incident_particles=np.tile(amplitude*polarization,(2,1)).astype(complex)
    alpha=sphere_polarizability(np.full(2,radius_nm*1e-9),wavelength_nm*1e-9,
                                silver_permittivity,medium_permittivity)
    k=2*np.pi*np.sqrt(medium_permittivity)/(wavelength_nm*1e-9)
    solved=solve_coupled_dipoles(centers,alpha,incident_particles,k,medium_permittivity)
    gap_position=np.zeros(3);scattered=np.zeros(3,dtype=complex)
    for center,dipole in zip(centers,solved.dipoles_c_m):
        scattered += electric_dyadic(gap_position-center,k,medium_permittivity) @ dipole
    incident=amplitude*polarization
    return DimerResult(gap_nm,radius_nm,wavelength_nm,polarization_deg,centers,gap_position,
                       incident,scattered,incident+scattered,solved.dipoles_c_m)
