import numpy as np
from stochastic_2d.geometry import GeometryConfig,generate_aggregate
from stochastic_2d.hotspots import construct_hotspots,hotspot_distances
from stochastic_2d.validation_level3a_5 import enhancement_classes,plasmon_remote

def test_unique_corrected_origin_and_reproducibility():
    h=construct_hotspots(generate_aggregate(GeometryConfig(n_particles=50,seed=4)))
    s1,d1=hotspot_distances(h,np.array([3]));s2,d2=hotspot_distances(h,np.array([3]))
    assert np.sum(d1==0)==1 and np.all(np.isfinite(d1)) and np.all(d1>=0)
    np.testing.assert_array_equal(s1,s2);np.testing.assert_array_equal(d1,d2)
def test_threshold_classes_are_nested():
    c=enhancement_classes([.9,1.01,2,10,100])
    assert c["any"].tolist()==[False,True,True,True,True]
    assert np.all(c["100x"]<=c["10x"]) and np.all(c["10x"]<=c["2x"])
def test_plasmon_remote_ratio_and_finite_statistics():
    mask,d=plasmon_remote(np.array([True,True,False]),np.array([1.,1.,.01]),np.array([99.,100.,1000.]))
    assert mask.tolist()==[False,True,False] and np.all(np.isfinite(d))
