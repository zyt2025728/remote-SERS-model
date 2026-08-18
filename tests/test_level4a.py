"""Focused invariants for the reusable Level 4A ensemble pipeline."""
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from stochastic_2d.geometry import GeometryConfig, generate_aggregate
from stochastic_2d.level4a import (BASE_GEOMETRY, ComplexGapCalibration,
    directed_transitions, equal_seed_mean, frozen_configuration, gamma_de,
    verify_level3b)


def test_frozen_configuration_and_only_seed_changes():
    frozen = frozen_configuration()
    assert frozen["only_variable_parameter"] == "random_seed"
    assert "seed" not in frozen["geometry"]
    a = asdict(replace(BASE_GEOMETRY, seed=1)); b = asdict(replace(BASE_GEOMETRY, seed=2))
    assert {key for key in a if a[key] != b[key]} == {"seed"}


def test_morphology_is_deterministic_but_seed_sensitive():
    config = GeometryConfig(n_particles=20, seed=919)
    first = generate_aggregate(config); replay = generate_aggregate(config)
    other = generate_aggregate(replace(config, seed=920))
    np.testing.assert_array_equal(first.positions_nm, replay.positions_nm)
    assert not np.array_equal(first.positions_nm, other.positions_nm)


def test_complex_calibration_and_no_extrapolation():
    calibration = ComplexGapCalibration()
    cp, ct, valid = calibration.interpolate(np.array([1., 1.25, 6., .99, 6.01]))
    assert np.iscomplexobj(cp) and np.iscomplexobj(ct)
    assert np.all(np.isfinite(cp[:3])) and valid[:3].all()
    assert np.isnan(cp[3:].real).all() and not valid[3:].any()
    assert cp[1].imag != 0 and ct[1].imag != 0


def test_fixed_classification_and_downstream_direction():
    # The lower geodesic endpoint is upstream even when adjacency is reversed.
    assert directed_transitions(((1,), (0, 2), (1,)), np.array([20., 10., 30.])) == [(1, 0), (1, 2)]
    np.testing.assert_allclose(gamma_de([2., 4.], [8., 2.]), [4., .5])
    frozen = frozen_configuration()
    assert frozen["remote_region_definition"].endswith("> 500 nm")
    assert frozen["geodesic_bin_edges_nm"] == list(np.arange(0., 2100., 100.))


def test_realization_balancing_gives_seeds_equal_weight():
    frame = pd.DataFrame({"seed": [1, 1, 1, 2], "value": [0., 0., 0., 10.]})
    assert frame.value.mean() == 2.5
    assert equal_seed_mean(frame, "value") == 5.0


def test_level3b_provenance_and_no_miepy_sweep_dependency():
    verify_level3b()
    source = Path("stochastic_2d/level4a.py").read_text()
    assert "run_level3b" not in source
    assert "Egap_MIEPY_dimer" not in source
