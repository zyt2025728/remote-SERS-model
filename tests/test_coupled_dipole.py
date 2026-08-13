import numpy as np

from stochastic_2d.coupled_dipole import assemble_matrix, scattered_field, solve_coupled_dipoles
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
    scattered = scattered_field(positions, first.dipoles_c_m, k)
    np.testing.assert_allclose(first.dipoles_c_m, alpha[:, None] * (field + scattered), rtol=1e-12)


def test_matrix_uses_non_neighbor_pair_interactions():
    positions = np.array([[0., 0., 0.], [50e-9, 0., 0.], [150e-9, 0., 0.]])
    alpha = np.full(3, 1e-34 + 1e-35j)
    matrix = assemble_matrix(positions, alpha, 2 * np.pi / 633e-9)
    # Particles 0 and 2 would not be nearest graph neighbors, but their full
    # 3x3 electromagnetic coupling block is present and complex.
    block = matrix[0:3, 6:9]
    assert np.any(block != 0)
    assert np.iscomplexobj(block)


def test_localized_source_decays_away_from_center():
    positions = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    field = gaussian_incident_field(positions, positions[0], 1.0)
    assert abs(field[0, 0]) > abs(field[1, 0]) > abs(field[2, 0])
    assert np.all(field[:, 1:] == 0)
