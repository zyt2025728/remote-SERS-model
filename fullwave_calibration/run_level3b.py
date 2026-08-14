"""Generate PDA reference sweep and an honest pending Level-3B manifest."""
from __future__ import annotations
import csv,shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .pda_dimer import solve_pda_dimer

GAPS=(1.,1.5,2.,2.5,3.,4.,5.,6.);ANGLES=(0.,30.,60.,90.)
def _write(path,header,rows):
    with path.open("w",newline="") as f:w=csv.writer(f,lineterminator="\n");w.writerow(header);w.writerows(rows)
def run(output_dir="results/level3B"):
    out=Path(output_dir);data=out/"data";data.mkdir(parents=True,exist_ok=True)
    comsol=shutil.which("comsol") or shutil.which("comsolbatch")
    rows=[]
    for gap in GAPS:
      for angle in ANGLES:
        r=solve_pda_dimer(gap,polarization_deg=angle);e=r.total;ep=e[0]
        rows.append([gap,10.,633.,angle,"PDA",e[0].real,e[0].imag,e[1].real,e[1].imag,e[2].real,e[2].imag,ep.real,ep.imag,np.linalg.norm(e),abs(ep),np.angle(ep,deg=True),1.,abs(ep)])
    header=["gap_nm","radius_nm","wavelength_nm","polarization_deg","solver_backend","Ex_real","Ex_imag","Ey_real","Ey_imag","Ez_real","Ez_imag","Eparallel_real","Eparallel_imag","Eabs","Eparallel_abs","Eparallel_phase_deg","E0","Eparallel_over_E0"]
    _write(data/"Egap_PDA_dimer.csv",header,rows)
    _write(data/"solver_status.csv",["item","value"],[
      ["COMSOL_executable",comsol or "not detected"],
      ["COMSOL_license","not testable; runtime absent" if not comsol else "must be checked by actual solve"],
      ["independent_fullwave_backend","not available"],
      ["solver_backend","PENDING_FULLWAVE"],
      ["calibration_status","blocked_pending_validated_fullwave_execution"],
      ["network_correction_applied","False"],
      ["frozen_radius_nm",10.],["frozen_wavelength_nm",633.],
      ["frozen_medium_refractive_index",1.],
      ["frozen_silver_permittivity","(-15+1j), unsourced illustrative Level-3A.5 assumption"],
      ["phasor_convention","exp(-i omega t)"],
    ])
    # Header-only contracts cannot be mistaken for numerical calibration.
    _write(data/"gap_calibration_table.csv",["gap_nm","radius_nm","wavelength_nm","polarization_deg","PDA_Eparallel_real","PDA_Eparallel_imag","FW_Eparallel_real","FW_Eparallel_imag","Cparallel_real","Cparallel_imag","Cparallel_abs","Cparallel_phase_deg","C_M2","C_M4","solver_backend","convergence_status"],[])
    _write(out/"USABLE_EGAP_TABLE.csv",["gap_nm","radius_nm","wavelength_nm","polarization_deg","solver_backend","Egap_complex_real_V_per_m","Egap_complex_imag_V_per_m","Egap_abs_V_per_m","Egap_phase_deg","Egap_over_E0_abs","PDA_Egap_abs_V_per_m","correction_abs","correction_phase_deg","convergence_status"],[])
    fig,ax=plt.subplots()
    for angle in ANGLES:
      selected=[x for x in rows if x[3]==angle];ax.plot([x[0] for x in selected],[x[14] for x in selected],"o-",label=f"{angle:g}°")
    ax.set(xlabel="gap (nm)",ylabel="|PDA Eparallel| / E0",title="Isolated-dimer PDA reference (not full-wave)");ax.legend();fig.tight_layout();fig.savefig(out/"Egap_PDA_vs_gap.svg",format="svg");plt.close(fig)
    return {"COMSOL_available":bool(comsol),"fullwave_backend":"unavailable","PDA_cases":len(rows),"calibration_rows":0,"network_correction_applied":False}
if __name__=="__main__":
    for k,v in run().items():print(f"{k}: {v}")
