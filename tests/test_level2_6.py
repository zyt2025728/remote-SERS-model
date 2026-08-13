import numpy as np
from stochastic_2d.geometry import GeometryConfig, generate_aggregate, minimum_surface_gap


def test_primary_geometry_has_monodisperse_r10_and_controlled_gaps():
    aggregate = generate_aggregate(GeometryConfig(n_particles=80, seed=26))
    np.testing.assert_array_equal(aggregate.radii_nm, np.full(80, 10.0))
    assert minimum_surface_gap(aggregate) >= 1.0 - 1e-9
    assert aggregate.edge_gaps_nm.min() >= 1.0 - 1e-9
    assert aggregate.edge_gaps_nm.max() <= 6.0 + 1e-9
    assert len(aggregate.edges) >= 79
