"""Complex vector coupled-dipole matrix assembly and solution."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .green_tensor import electric_dyadic


@dataclass(frozen=True)
class DipoleSolution:
    dipoles_c_m: NDArray[np.complex128]
    matrix: NDArray[np.complex128]
    residual_relative: float


def scattered_field(
    positions_m: NDArray[np.float64],
    dipoles_c_m: NDArray[np.complex128],
    wave_number: complex,
    medium_relative_permittivity: complex = 1.0,
) -> NDArray[np.complex128]:
    """Evaluate the field scattered at every center by all *other* dipoles.

    This intentionally uses every ordered particle pair, independently of the
    geometry graph. The graph describes morphology only and never truncates the
    electromagnetic interaction.
    """
    positions = np.asarray(positions_m, dtype=float)
    dipoles = np.asarray(dipoles_c_m, dtype=np.complex128)
    if positions.shape != dipoles.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions and dipoles must both have shape (N, 3)")
    field = np.zeros_like(dipoles)
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                field[i] += electric_dyadic(
                    positions[i] - positions[j], wave_number,
                    medium_relative_permittivity,
                ) @ dipoles[j]
    return field


def assemble_matrix(positions_m: NDArray[np.float64], polarizabilities: NDArray[np.complex128],
                    wave_number: complex, medium_relative_permittivity: complex = 1.0) -> NDArray[np.complex128]:
    positions = np.asarray(positions_m, dtype=float)
    n = len(positions)
    if positions.shape != (n, 3) or np.asarray(polarizabilities).shape != (n,):
        raise ValueError("positions must be (N,3) and polarizabilities must be (N,)")
    matrix = np.eye(3 * n, dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                green = electric_dyadic(positions[i] - positions[j], wave_number,
                                        medium_relative_permittivity)
                matrix[3*i:3*i+3, 3*j:3*j+3] = -polarizabilities[i] * green
    return matrix


def solve_coupled_dipoles(positions_m: NDArray[np.float64],
                          polarizabilities: NDArray[np.complex128],
                          incident_field: NDArray[np.complex128], wave_number: complex,
                          medium_relative_permittivity: complex = 1.0) -> DipoleSolution:
    """Solve p_i = alpha_i(E_inc + sum G_ij p_j) in complex arithmetic."""
    matrix = assemble_matrix(positions_m, polarizabilities, wave_number,
                             medium_relative_permittivity)
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("coupled-dipole matrix contains NaN or infinity")
    rhs = (np.asarray(polarizabilities)[:, None] * np.asarray(incident_field)).reshape(-1)
    solution = np.linalg.solve(matrix, rhs)
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("coupled-dipole solution contains NaN or infinity")
    residual = np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    return DipoleSolution(solution.reshape(-1, 3), matrix, float(residual))
