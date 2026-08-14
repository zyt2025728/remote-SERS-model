"""Level-3B PDA comparison and frozen-network correction (no Maxwell solves).

This module only consumes frozen CSV inputs.  It never imports or invokes the
MiePy solver, regenerates a stochastic geometry, or extrapolates calibration
factors beyond their tabulated gap range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

GAPS = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6], dtype=float)
ANGLES = np.array([0, 30, 60, 90], dtype=float)
KEYS = ["gap_nm", "radius_nm", "wavelength_nm", "polarization_deg"]
MIEPY_BACKEND = "miepy_GMMT"


def _complex(frame: pd.DataFrame, real: str, imag: str) -> np.ndarray:
    return frame[real].to_numpy(float) + 1j * frame[imag].to_numpy(float)


def interpolate_complex(gaps, values, requested):
    gaps, values, requested = np.asarray(gaps), np.asarray(values), np.asarray(requested)
    if np.any(requested < gaps.min()) or np.any(requested > gaps.max()):
        raise ValueError(f"gap outside calibrated [{gaps.min():g}, {gaps.max():g}] nm range")
    return np.interp(requested, gaps, values.real) + 1j*np.interp(requested, gaps, values.imag)


def load_dimer_inputs(miepy_path: Path, pda_path: Path) -> pd.DataFrame:
    if not pda_path.is_file():
        raise FileNotFoundError(
            f"Frozen PDA input is missing: {pda_path}. Refusing to synthesize PDA data."
        )
    fw, pda = pd.read_csv(miepy_path), pd.read_csv(pda_path)
    required_fw = {*KEYS, "solver_backend", "Egap_Ex_real_V_per_m", "Egap_Ex_imag_V_per_m",
                   "Egap_Ey_real_V_per_m", "Egap_Ey_imag_V_per_m", "convergence_status"}
    required_pda = {*KEYS, "PDA_Ex_real", "PDA_Ex_imag", "PDA_Ey_real", "PDA_Ey_imag"}
    for label, frame, required in (("MiePy", fw, required_fw), ("PDA", pda, required_pda)):
        missing = required - set(frame)
        if missing: raise ValueError(f"{label} table missing columns: {sorted(missing)}")
        if frame.duplicated(KEYS).any(): raise ValueError(f"duplicate physical rows in {label} table")
    if not fw.solver_backend.eq(MIEPY_BACKEND).all() or not fw.convergence_status.eq("PASS").all():
        raise ValueError("MiePy input must contain only converged miepy_GMMT production rows")
    merged = fw.merge(pda, on=KEYS, validate="one_to_one", suffixes=("_FW", "_PDA"))
    expected = pd.MultiIndex.from_product([GAPS, ANGLES], names=["gap_nm", "polarization_deg"])
    actual = pd.MultiIndex.from_frame(merged[["gap_nm", "polarization_deg"]])
    if len(merged) != 32 or set(actual) != set(expected):
        raise ValueError("dimer tables must provide exactly the same 8 gaps x 4 angles")
    return merged.sort_values(["gap_nm", "polarization_deg"]).reset_index(drop=True)


def build_calibration(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fw_x = _complex(rows, "Egap_Ex_real_V_per_m", "Egap_Ex_imag_V_per_m")
    fw_y = _complex(rows, "Egap_Ey_real_V_per_m", "Egap_Ey_imag_V_per_m")
    pda_x = _complex(rows, "PDA_Ex_real", "PDA_Ex_imag")
    pda_y = _complex(rows, "PDA_Ey_real", "PDA_Ey_imag")
    longitudinal, transverse = rows.polarization_deg.eq(0), rows.polarization_deg.eq(90)
    if np.any(np.abs(pda_x[longitudinal]) < 1e-14) or np.any(np.abs(pda_y[transverse]) < 1e-14):
        raise ValueError("PDA reference component is numerically zero")
    cp = fw_x[longitudinal] / pda_x[longitudinal]
    ct = fw_y[transverse] / pda_y[transverse]
    calibration = pd.DataFrame({"gap_nm": rows.loc[longitudinal, "gap_nm"].to_numpy(),
        "Cparallel_real": cp.real, "Cparallel_imag": cp.imag, "Cparallel_abs": abs(cp),
        "Cparallel_phase_deg": np.angle(cp, deg=True), "Cperp_real": ct.real,
        "Cperp_imag": ct.imag, "Cperp_abs": abs(ct), "Cperp_phase_deg": np.angle(ct, deg=True),
        "C_M2_parallel": abs(cp)**2, "C_M4_parallel": abs(cp)**4,
        "C_M2_perp": abs(ct)**2, "C_M4_perp": abs(ct)**4})
    validation = []
    for angle in (30, 60):
        mask = rows.polarization_deg.eq(angle)
        pred_x = cp * pda_x[mask]; pred_y = ct * pda_y[mask]
        pred_abs = np.sqrt(abs(pred_x)**2 + abs(pred_y)**2)
        actual_abs = np.sqrt(abs(fw_x[mask])**2 + abs(fw_y[mask])**2)
        # Compare phase of the component projected onto incident polarization.
        rad = np.deg2rad(angle)
        pred_proj, actual_proj = pred_x*np.cos(rad)+pred_y*np.sin(rad), fw_x[mask]*np.cos(rad)+fw_y[mask]*np.sin(rad)
        validation.extend(pd.DataFrame({"gap_nm": rows.loc[mask,"gap_nm"], "polarization_deg": angle,
            "predicted_abs": pred_abs, "actual_abs": actual_abs,
            "relative_magnitude_error": abs(pred_abs-actual_abs)/actual_abs,
            "phase_error_deg": abs((np.angle(pred_proj/actual_proj,deg=True)+180)%360-180)}).to_dict("records"))
    return calibration, pd.DataFrame(validation)


def correct_network(network: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    required = {"hotspot_id", "gap_nm", "u_gap_x", "u_gap_y", "PDA_Ex_real", "PDA_Ex_imag",
                "PDA_Ey_real", "PDA_Ey_imag", "remote_classification", "x_nm", "y_nm"}
    missing = required-set(network)
    if missing: raise ValueError(f"frozen network table missing columns: {sorted(missing)}")
    out = network.loc[network.gap_nm.between(GAPS.min(), GAPS.max())].copy()
    u = out[["u_gap_x","u_gap_y"]].to_numpy(float)
    norms = np.linalg.norm(u,axis=1)
    if not np.allclose(norms,1,rtol=0,atol=1e-10): raise ValueError("local gap axes must be unit vectors")
    E = np.column_stack((_complex(out,"PDA_Ex_real","PDA_Ex_imag"), _complex(out,"PDA_Ey_real","PDA_Ey_imag")))
    projected = np.sum(E*u,axis=1); Epar=projected[:,None]*u; Eperp=E-Epar
    cp0=_complex(calibration,"Cparallel_real","Cparallel_imag"); ct0=_complex(calibration,"Cperp_real","Cperp_imag")
    cp=interpolate_complex(calibration.gap_nm,cp0,out.gap_nm); ct=interpolate_complex(calibration.gap_nm,ct0,out.gap_nm)
    corrected=cp[:,None]*Epar+ct[:,None]*Eperp
    for axis,k in (("Ex",0),("Ey",1)):
        out[f"FWcorrected_{axis}_real"]=corrected[:,k].real; out[f"FWcorrected_{axis}_imag"]=corrected[:,k].imag
    out["Cparallel_real"],out["Cparallel_imag"]=cp.real,cp.imag; out["Cperp_real"],out["Cperp_imag"]=ct.real,ct.imag
    out["M2_PDA"]=np.sum(abs(E)**2,axis=1); out["M4_PDA"]=out.M2_PDA**2
    out["M2_corrected"]=np.sum(abs(corrected)**2,axis=1); out["M4_corrected"]=out.M2_corrected**2
    return out


def _lineplot(path, groups, ylabel, ratio=False):
    fig, axes=plt.subplots(2 if ratio else 1,sharex=True,figsize=(6,6 if ratio else 4)); axes=np.atleast_1d(axes)
    for label,x,y in groups: axes[0].plot(x,y,"o-",label=label)
    axes[0].set_ylabel(ylabel); axes[0].legend()
    if ratio: axes[1].plot(groups[0][1],groups[0][2]/groups[1][2],"o-"); axes[1].set_ylabel("MiePy/PDA")
    axes[-1].set_xlabel("gap (nm)"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def write_outputs(rows, calibration, validation, network, output: Path):
    data=output/"data"; data.mkdir(parents=True,exist_ok=True)
    calibration.to_csv(data/"gap_calibration_table.csv",index=False); validation.to_csv(data/"component_correction_validation.csv",index=False)
    lon=rows[rows.polarization_deg.eq(0)]; fw=abs(_complex(lon,"Egap_Ex_real_V_per_m","Egap_Ex_imag_V_per_m")); pda=abs(_complex(lon,"PDA_Ex_real","PDA_Ex_imag"))
    _lineplot(output/"PDA_vs_MIEPY_longitudinal.svg",[("MiePy",lon.gap_nm,fw),("PDA",lon.gap_nm,pda)],"|Egap/E0|",True)
    for c,name,ylabel in (("Cparallel_abs","Cparallel_magnitude_vs_gap.svg","|Cparallel|"),("Cparallel_phase_deg","Cparallel_phase_vs_gap.svg","phase(Cparallel) (deg)"),("Cperp_abs","Cperp_magnitude_vs_gap.svg","|Cperp|"),("Cperp_phase_deg","Cperp_phase_vs_gap.svg","phase(Cperp) (deg)")):
        _lineplot(output/name,[(c,calibration.gap_nm,calibration[c])],ylabel)
    fig,ax=plt.subplots();
    for a,g in validation.groupby("polarization_deg"): ax.plot(g.gap_nm,100*g.relative_magnitude_error,"o-",label=f"{a:g} deg magnitude")
    ax.set(xlabel="gap (nm)",ylabel="relative error (%)");ax.legend();fig.tight_layout();fig.savefig(output/"component_correction_validation.svg");plt.close(fig)
    network.to_csv(data/"network_hotspots_fullwave_corrected.csv",index=False)
    for col,name in (("M4_PDA","network_M4_PDA.svg"),("M4_corrected","network_M4_fullwave_corrected.svg")):
        fig,ax=plt.subplots(); sc=ax.scatter(network.x_nm,network.y_nm,c=np.log10(network[col]),s=18);fig.colorbar(sc,ax=ax,label=f"log10({col})");ax.set_aspect("equal");fig.tight_layout();fig.savefig(output/name);plt.close(fig)
    remote=network[network.remote_classification.astype(str).str.lower().eq("remote")].copy()
    remote["rank_PDA"]=remote.M4_PDA.rank(ascending=False);remote["rank_corrected"]=remote.M4_corrected.rank(ascending=False)
    fig,ax=plt.subplots();ax.scatter(remote.rank_PDA,remote.rank_corrected);ax.plot([1,len(remote)],[1,len(remote)],"--");ax.set(xlabel="PDA rank",ylabel="corrected rank");fig.tight_layout();fig.savefig(output/"remote_hotspot_ranking_before_after.svg");plt.close(fig)
    strongest=remote.nlargest(1,"M4_corrected");fig,ax=plt.subplots();ax.scatter(network.x_nm,network.y_nm,s=5);ax.scatter(strongest.x_nm,strongest.y_nm,s=80,marker="*");ax.set_aspect("equal");fig.tight_layout();fig.savefig(output/"strongest_corrected_remote_hotspot.svg");plt.close(fig)
    top_a=set(remote.nlargest(10,"M4_PDA").hotspot_id);top_b=set(remote.nlargest(10,"M4_corrected").hotspot_id)
    source=network[network.remote_classification.astype(str).str.lower().eq("source")]
    summary=pd.DataFrame([{"strongest_PDA_remote_hotspot_id":remote.nlargest(1,"M4_PDA").hotspot_id.iloc[0],"strongest_corrected_remote_hotspot_id":strongest.hotspot_id.iloc[0],"same_gap":remote.nlargest(1,"M4_PDA").hotspot_id.iloc[0]==strongest.hotspot_id.iloc[0],"top_10_overlap":len(top_a&top_b),"spearman_rank_correlation":spearmanr(remote.rank_PDA,remote.rank_corrected).statistic,"source_max_M4":source.M4_corrected.max(),"remote_max_corrected_M4":strongest.M4_corrected.iloc[0],"corrected_remote_source_M4_ratio":strongest.M4_corrected.iloc[0]/source.M4_corrected.max()}])
    summary.to_csv(data/"network_before_after_summary.csv",index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument("--miepy",type=Path,default=Path("results/level3B/USABLE_EGAP_TABLE.csv"));p.add_argument("--pda",type=Path,default=Path("results/level3B/data/Egap_PDA_dimer.csv"));p.add_argument("--network",type=Path,default=Path("results/level3A5/data/network_hotspots.csv"));p.add_argument("--output",type=Path,default=Path("results/level3B"));a=p.parse_args()
    rows=load_dimer_inputs(a.miepy,a.pda);cal,val=build_calibration(rows)
    if val.relative_magnitude_error.max()>0.05 or val.phase_error_deg.max()>2: raise RuntimeError("component reconstruction validation failed; network correction refused")
    if not a.network.is_file(): raise FileNotFoundError(f"Frozen network input is missing: {a.network}. Refusing to regenerate geometry.")
    net=correct_network(pd.read_csv(a.network),cal);write_outputs(rows,cal,val,net,a.output)


if __name__ == "__main__": main()
