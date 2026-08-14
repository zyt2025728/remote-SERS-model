import csv
import numpy as np
import pytest
from fullwave_calibration.calibration import FullWaveCalibration,FullWaveCalibrationError
from fullwave_calibration.pda_dimer import solve_pda_dimer

def test_pda_dimer_geometry_and_complex_solution():
    r=solve_pda_dimer(2.,polarization_deg=30)
    assert np.isclose(np.linalg.norm(r.centers_m[1]-r.centers_m[0])-20e-9,2e-9)
    np.testing.assert_array_equal(r.gap_position_m,np.zeros(3))
    assert np.iscomplexobj(r.total) and np.all(np.isfinite(r.total))
def test_completed_component_table_is_populated_and_complex():
    import pandas as pd
    table=pd.read_csv("results/level3B/data/gap_calibration_table.csv")
    assert len(table)==8
    assert set(table.gap_nm)=={1,1.5,2,2.5,3,4,5,6}
    assert np.all(np.isfinite(table[["Cparallel_real","Cparallel_imag","Cperp_real","Cperp_imag"]]))
def test_interpolation_exact_and_bounds(tmp_path):
    p=tmp_path/"c.csv";header=FullWaveCalibration.REQUIRED
    with p.open("w",newline="") as f:
      w=csv.writer(f);w.writerow(header)
      for g in (1.,2.):
       for angle in (0.,90.):w.writerow([g,10.,633.,angle,g+angle/100,1.,"VALIDATED_MAXWELL_TEST_FIXTURE","converged"])
    c=FullWaveCalibration(p)
    assert c.get_complex_gap_correction(1.,10.,633.,0.)==1+1j
    with pytest.raises(FullWaveCalibrationError,match="outside"):
        c.get_complex_gap_correction(.5,10.,633.,0.)
