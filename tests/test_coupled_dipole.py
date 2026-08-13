import numpy as np

from stochastic_2d.coupled_dipole import assemble_matrix, solve_coupled_dipoles
from stochastic_2d.excitation import gaussian_incident_field
from stochastic_2d.geometry import GeometryConfig, generate_aggregate
from stochastic_2d.polarizability import sphere_polarizability


def test_complex_solver_matrix_and_solution_are_finite_and_reproducible():
    aggregate = generate_aggregate(GeometryConfig(n_particles=12, seed=12))
    positions = np.column_stack((aggregate.positions_nm, np.zeros(12))) * 1e-9
    wavelength = 633e-9
    alpha = sphere_polarizability(aggregate.radii_nm * 1e-9, wavelength, -15 + 1j)
    field = gaussian_incident_field(positions, positions[0], 60e-9,
                                    amplitude_v_per_m=1 + 0.25j)
    k = 2 * np.pi / wavelength
    matrix = assemble_matrix(positions, alpha, k)
    first = solve_coupled_dipoles(positions, alpha, field, k)
    second = solve_coupled_dipoles(positions, alpha, field, k)
    assert np.iscomplexobj(matrix)
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(first.dipoles_c_m))
    assert first.residual_relative < 1e-10
    np.testing.assert_array_equal(first.dipoles_c_m, second.dipoles_c_m)


def test_localized_source_decays_away_from_center():
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    field = gaussian_incident_field(positions, positions[0], 1.0)
    assert abs(field[0, 0]) > abs(field[1, 0]) > abs(field[2, 0])
    assert np.all(field[:, 1:] == 0)
