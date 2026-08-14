import numpy as np

from stochastic_2d.geometry import GeometryConfig, generate_aggregate, minimum_surface_gap


def connected_component_count(aggregate):
    adjacency = [[] for _ in range(aggregate.n_particles)]
    for i, j in aggregate.edges:
        adjacency[i].append(j); adjacency[j].append(i)
    seen = {0}; stack = [0]
    while stack:
        stack.extend(node for node in adjacency[stack.pop()] if node not in seen and not seen.add(node))
    return 1 + int(len(seen) != aggregate.n_particles)


def test_geometry_is_reproducible_connected_and_nonoverlapping():
    config = GeometryConfig(n_particles=60, seed=8128)
    first = generate_aggregate(config)
    second = generate_aggregate(config)
    np.testing.assert_array_equal(first.positions_nm, second.positions_nm)
    np.testing.assert_array_equal(first.radii_nm, second.radii_nm)
    np.testing.assert_array_equal(first.edges, second.edges)
    assert minimum_surface_gap(first) >= -1e-8
    assert connected_component_count(first) == 1


def test_disordered_topology_has_variable_coordination_and_dead_ends():
    aggregate = generate_aggregate(GeometryConfig(n_particles=100, seed=42))
    assert len(np.unique(aggregate.degrees)) >= 3
    assert np.count_nonzero(aggregate.degrees == 1) > 0
    assert len(aggregate.edges) >= aggregate.n_particles - 1
    assert np.std(aggregate.positions_nm[:, 0]) > 0
    assert np.std(aggregate.positions_nm[:, 1]) > 0
