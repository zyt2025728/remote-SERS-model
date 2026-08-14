import numpy as np
import pytest

from fullwave_calibration.calibration import CONFIG, converged, phase_delta_deg
from fullwave_calibration.import_comsol_results import EXPECTED_EPSILON, validate
import pandas as pd
import miepy


def test_exact_geometry():
    for gap in CONFIG.gaps_nm:
        centers = CONFIG.centers(gap)
        assert np.isclose(np.linalg.norm(centers[1]-centers[0])/1e-9, 2*CONFIG.radius_nm+gap)
        assert np.array_equal((centers[0]+centers[1])/2, CONFIG.gap_center)


def test_convergence_classifier():
    assert converged(0.0049, 0.49)
    assert not converged(0.005, 0.49)
    assert not converged(0.0049, 0.5)
    assert phase_delta_deg(179, -179) == 2


def test_no_extrapolation_contract():
    def interpolate(gap):
        if not min(CONFIG.gaps_nm) <= gap <= max(CONFIG.gaps_nm):
            raise ValueError("outside calibrated range")
        return np.interp(gap, CONFIG.gaps_nm, CONFIG.gaps_nm)
    for gap in CONFIG.gaps_nm:
        assert interpolate(gap) == gap
    with pytest.raises(ValueError):
        interpolate(0.9)


def test_material_is_exact_johnson_value():
    actual = complex(miepy.materials.Ag(author="Johnson").eps(CONFIG.wavelength_nm * 1e-9))
    assert actual == EXPECTED_EPSILON


def test_comsol_import_rejects_incompatible_physics():
    row = dict(gap_nm=3, radius_nm=10, wavelength_nm=633, polarization_deg=0,
        epsilon_Ag_real=EXPECTED_EPSILON.real, epsilon_Ag_imag=EXPECTED_EPSILON.imag,
        E0_V_per_m=1, convergence_status="PASS", Ex_real=1, Ex_imag=0,
        Ey_real=0, Ey_imag=0, Ez_real=0, Ez_imag=0)
    validate(pd.DataFrame([row]))
    row["wavelength_nm"] = 532
    with pytest.raises(ValueError, match="wavelength"):
        validate(pd.DataFrame([row]))
