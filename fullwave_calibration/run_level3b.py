"""Generate the matched 32-case PDA reference (never invokes MiePy)."""
from __future__ import annotations
import csv
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
    rows=[]
    for gap in GAPS:
      for angle in ANGLES:
        r=solve_pda_dimer(gap,polarization_deg=angle);e=r.total
        ep=e[0]; et=e[1]
        rows.append([gap,10.,633.,angle,"PDA",e[0].real,e[0].imag,e[1].real,e[1].imag,e[2].real,e[2].imag,ep.real,ep.imag,et.real,et.imag,*sum(([abs(z),np.angle(z,deg=True)] for z in (*e,ep,et)),[])])
    header=["gap_nm","radius_nm","wavelength_nm","polarization_deg","solver_backend","Ex_real","Ex_imag","Ey_real","Ey_imag","Ez_real","Ez_imag","Eparallel_real","Eparallel_imag","Eperp_real","Eperp_imag","Ex_abs","Ex_phase_deg","Ey_abs","Ey_phase_deg","Ez_abs","Ez_phase_deg","Eparallel_abs","Eparallel_phase_deg","Eperp_abs","Eperp_phase_deg"]
    _write(data/"Egap_PDA_dimer.csv",header,rows)
    fig,ax=plt.subplots()
    for angle in ANGLES:
      selected=[x for x in rows if x[3]==angle];ax.plot([x[0] for x in selected],[x[21] for x in selected],"o-",label=f"{angle:g}°")
    ax.set(xlabel="gap (nm)",ylabel="|PDA Eparallel| / E0",title="Isolated-dimer PDA reference (not full-wave)");ax.legend();fig.tight_layout();fig.savefig(out/"Egap_PDA_vs_gap.svg",format="svg");plt.close(fig)
    return {"solver_backend":"PDA matched reference","PDA_cases":len(rows)}
if __name__=="__main__":
    for k,v in run().items():print(f"{k}: {v}")
