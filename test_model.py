import numpy as np

from run_2d_sers_model import N_COLS, N_ROWS, run_simulation


def test_representative_simulation_is_finite_and_attenuates():
    particles, propagation, summary = run_simulation()
    assert len(particles) == N_ROWS * N_COLS
    assert np.isfinite(particles.select_dtypes(include=[np.number])).all().all()
    assert summary["matrix_condition_number"] < 1e8
    assert summary["attenuation_coefficient_per_nm"] > 0
    assert 0 <= summary["fit_r_squared"] <= 1
    assert propagation.relative_intensity.iloc[-1] < propagation.relative_intensity.iloc[0]
