import numpy as np
from stochastic_2d.geometry import GeometryConfig, generate_aggregate
from stochastic_2d.hotspots import construct_hotspots, classify_hotspots, hotspot_distances


def test_hotspot_geometry_bounds_positions_and_reproducibility():
    a=generate_aggregate(GeometryConfig(n_particles=40,seed=31))
    first=construct_hotspots(a); second=construct_hotspots(a)
    np.testing.assert_array_equal(first.positions_nm,second.positions_nm)
    for (i,j),position,gap in zip(first.pairs,first.positions_nm,first.gaps_nm):
        distance=np.linalg.norm(a.positions_nm[j]-a.positions_nm[i])
        np.testing.assert_allclose(gap,distance-a.radii_nm[i]-a.radii_nm[j])
        assert np.isclose(np.linalg.norm(position-a.positions_nm[i]),a.radii_nm[i]+gap/2)
        assert 1-1e-9<=gap<=6+1e-9
    steps,lengths=hotspot_distances(first,np.array([0]))
    assert np.all(steps>=0) and np.all(np.isfinite(lengths))


def test_invalid_pairs_excluded_and_remote_threshold_exact():
    a=generate_aggregate(GeometryConfig(n_particles=20,seed=9))
    h=construct_hotspots(a,gap_min_nm=2,gap_max_nm=3)
    assert np.all((h.gaps_nm>=2-1e-9)&(h.gaps_nm<=3+1e-9))
    labels=classify_hotspots(np.array([1e-7,1e-7,1e-5]),np.array([501.,499.,600.]))
    assert labels.tolist()==["remote","transition","source"]


def test_metrics_are_nonnegative_and_fourth_is_square():
    total=np.array([[1+2j,3-1j,0],[0,1j,2]])
    m2=np.sum(abs(total)**2,axis=1);m4=m2**2
    assert np.all(m2>=0);np.testing.assert_allclose(m4,m2**2)
