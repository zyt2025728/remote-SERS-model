import numpy as np
import pytest

from fullwave_calibration.calibration import CONFIG, converged, phase_delta_deg
from fullwave_calibration.import_comsol_results import EXPECTED_EPSILON, validate
from fullwave_calibration.postprocess import build_calibration, correct_network, interpolate_complex
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


def test_complex_interpolation_nodes_and_no_extrapolation():
    values = np.arange(8) + 1j*np.arange(8)[::-1]
    assert np.array_equal(interpolate_complex(CONFIG.gaps_nm, values, CONFIG.gaps_nm), values)
    with pytest.raises(ValueError, match="outside calibrated"):
        interpolate_complex(CONFIG.gaps_nm, values, [6.01])


def test_no_longitudinal_division_at_90_and_phase_preserved():
    rows=[]
    cp, ct=2+1j, -0.5+0.25j
    for gap in CONFIG.gaps_nm:
        for angle in CONFIG.polarizations_deg:
            t=np.deg2rad(angle); px=np.cos(t)*(1+0.2j); py=np.sin(t)*(0.7-0.1j)
            rows.append(dict(gap_nm=gap,polarization_deg=angle,
                PDA_Ex_real=px.real,PDA_Ex_imag=px.imag,PDA_Ey_real=py.real,PDA_Ey_imag=py.imag,
                Egap_Ex_real_V_per_m=(cp*px).real,Egap_Ex_imag_V_per_m=(cp*px).imag,
                Egap_Ey_real_V_per_m=(ct*py).real,Egap_Ey_imag_V_per_m=(ct*py).imag))
    calibration, validation=build_calibration(pd.DataFrame(rows))
    assert np.allclose(calibration.Cparallel_real+1j*calibration.Cparallel_imag,cp)
    assert np.allclose(calibration.Cperp_real+1j*calibration.Cperp_imag,ct)
    assert validation.relative_magnitude_error.max()<1e-14
    network=pd.DataFrame([dict(hotspot_id="h",gap_nm=3,u_gap_x=1.,u_gap_y=0.,
        PDA_Ex_real=1.,PDA_Ex_imag=1.,PDA_Ey_real=0.,PDA_Ey_imag=0.,
        remote_classification="remote",x_nm=0.,y_nm=0.)])
    corrected=correct_network(network,calibration)
    got=corrected.FWcorrected_Ex_real.iloc[0]+1j*corrected.FWcorrected_Ex_imag.iloc[0]
    assert np.isclose(got,cp*(1+1j))
