"""Bounds-checked interpolation of independently solved full-wave/PDA corrections."""
from __future__ import annotations
import csv,itertools
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class FullWaveCalibrationError(RuntimeError): pass

class FullWaveCalibration:
    REQUIRED=("gap_nm","radius_nm","wavelength_nm","polarization_deg",
              "Cparallel_real","Cparallel_imag","solver_backend","convergence_status")
    def __init__(self,path):
        with Path(path).open(newline="") as f:reader=csv.DictReader(f);rows=list(reader)
        missing=set(self.REQUIRED)-set(reader.fieldnames or [])
        if missing:raise FullWaveCalibrationError(f"missing columns: {sorted(missing)}")
        if not rows:raise FullWaveCalibrationError("no executed full-wave calibration rows")
        if any(r["convergence_status"]!="converged" for r in rows):
            raise FullWaveCalibrationError("table contains non-converged full-wave rows")
        if any(r["solver_backend"] in ("","PENDING","PDA") for r in rows):
            raise FullWaveCalibrationError("an independent full-wave backend is required")
        axes=[sorted({float(r[k]) for r in rows}) for k in self.REQUIRED[:4]]
        lookup={(float(r["gap_nm"]),float(r["radius_nm"]),float(r["wavelength_nm"]),float(r["polarization_deg"])):complex(float(r["Cparallel_real"]),float(r["Cparallel_imag"])) for r in rows}
        expected=list(itertools.product(*axes))
        if any(p not in lookup for p in expected):raise FullWaveCalibrationError("calibration grid is incomplete")
        values=np.array([lookup[p] for p in expected]).reshape(*(len(x) for x in axes))
        self.axes=axes;self.interpolator=RegularGridInterpolator(axes,values,bounds_error=True)
    def get_complex_gap_correction(self,gap_nm,radius_nm,wavelength_nm,polarization_deg):
        try:return complex(self.interpolator([[gap_nm,radius_nm,wavelength_nm,polarization_deg]])[0])
        except ValueError as exc:raise FullWaveCalibrationError("query outside calibrated domain; extrapolation forbidden") from exc
