"""Run the 32-case interacting MiePy dimer calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import miepy
import numpy as np
import pandas as pd

NM = 1e-9
SOLVER_BACKEND = "miepy_GMMT"


@dataclass(frozen=True)
class Configuration:
    radius_nm: float = 10.0
    wavelength_nm: float = 633.0
    e0_v_per_m: float = 1.0
    gaps_nm: tuple[float, ...] = (1, 1.5, 2, 2.5, 3, 4, 5, 6)
    polarizations_deg: tuple[float, ...] = (0, 30, 60, 90)
    lmax_min: int = 1
    lmax_cap: int = 60
    magnitude_tolerance: float = 0.005
    phase_tolerance_deg: float = 0.5
    consecutive_checks: int = 2

    @property
    def gap_center(self) -> np.ndarray:
        return np.zeros(3)

    def centers(self, gap_nm: float) -> np.ndarray:
        half = (self.radius_nm + gap_nm / 2) * NM
        return np.array([[-half, 0, 0], [half, 0, 0]])


CONFIG = Configuration()


def phase_delta_deg(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def converged(mag_change: float, phase_change: float, config: Configuration = CONFIG) -> bool:
    return mag_change < config.magnitude_tolerance and phase_change < config.phase_tolerance_deg


def solve_case(gap_nm: float, angle_deg: float, config: Configuration = CONFIG):
    silver = miepy.materials.Ag(author="Johnson")
    medium = miepy.materials.air()
    theta = np.deg2rad(angle_deg)
    source = miepy.sources.plane_wave(
        polarization=[np.cos(theta), np.sin(theta)], amplitude=config.e0_v_per_m
    )
    history, successes, previous = [], 0, None
    for lmax in range(config.lmax_min, config.lmax_cap + 1):
        model = miepy.sphere_cluster(
            position=config.centers(gap_nm), radius=config.radius_nm * NM,
            material=silver, source=source, wavelength=config.wavelength_nm * NM,
            lmax=lmax, medium=medium, interactions=True, method=miepy.solver.bicgstab,
        )
        E = np.asarray(model.E_field(*config.gap_center), dtype=complex).reshape(3)
        # Projection on the incident-polarization direction is nonzero for
        # longitudinal, transverse, and mixed cases and carries a unique phase.
        relevant = E[0] * np.cos(theta) + E[1] * np.sin(theta)
        mag, phase = abs(relevant), np.angle(relevant, deg=True)
        dm = np.nan if previous is None else abs(mag - previous[0]) / mag
        dp = np.nan if previous is None else phase_delta_deg(phase, previous[1])
        successes = successes + 1 if previous is not None and converged(dm, dp, config) else 0
        status = "PASS" if successes >= config.consecutive_checks else "PENDING"
        history.append(dict(gap_nm=gap_nm, polarization_deg=angle_deg, lmax=lmax,
            Egap_real=relevant.real, Egap_imag=relevant.imag, Egap_abs=mag,
            Egap_phase_deg=phase, relative_magnitude_change=dm,
            phase_change_deg=dp, convergence_status=status))
        if status == "PASS":
            return E, history
        previous = (mag, phase)
    history[-1]["convergence_status"] = "FAIL"
    return None, history


def field_row(gap, angle, E, final, config=CONFIG):
    parallel = E[0]
    perp = complex(E[1])
    total = float(np.linalg.norm(E))
    row = dict(gap_nm=gap, radius_nm=config.radius_nm,
        wavelength_nm=config.wavelength_nm, polarization_deg=angle,
        solver_backend=SOLVER_BACKEND)
    for name, value in zip(("Ex", "Ey", "Ez"), E):
        row.update({f"Egap_{name}_real_V_per_m": value.real,
                    f"Egap_{name}_imag_V_per_m": value.imag,
                    f"Egap_{name}_abs_V_per_m": abs(value),
                    f"Egap_{name}_phase_deg": np.angle(value, deg=True)})
    row.update(Eparallel_real=parallel.real, Eparallel_imag=parallel.imag,
        Eparallel_abs=abs(parallel), Eparallel_phase_deg=np.angle(parallel, deg=True),
        Eperp_real=perp.real, Eperp_imag=perp.imag, Eperp_abs=abs(perp),
        Eperp_phase_deg=np.angle(perp, deg=True), Egap_total_abs=total,
        Egap_total_over_E0=total/config.e0_v_per_m,
        Eparallel_over_E0=abs(parallel)/config.e0_v_per_m,
        lmax_used=final["lmax"],
        magnitude_convergence_percent=100*final["relative_magnitude_change"],
        phase_convergence_deg=final["phase_change_deg"], convergence_status="PASS")
    return row


def run(output: Path = Path("results/level3B")) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = output / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    fields, histories = [], []
    for gap in CONFIG.gaps_nm:
        for angle in CONFIG.polarizations_deg:
            E, history = solve_case(gap, angle)
            histories.extend(history)
            if E is not None:
                fields.append(field_row(gap, angle, E, history[-1]))
    history_df, field_df = pd.DataFrame(histories), pd.DataFrame(fields)
    history_df.to_csv(data_dir / "miepy_multipole_convergence.csv", index=False)
    field_df.to_csv(data_dir / "Egap_MIEPY_dimer.csv", index=False)
    usable_columns = ["gap_nm", "radius_nm", "wavelength_nm", "polarization_deg", "solver_backend",
        "Egap_Ex_real_V_per_m", "Egap_Ex_imag_V_per_m", "Egap_Ey_real_V_per_m",
        "Egap_Ey_imag_V_per_m", "Egap_Ez_real_V_per_m", "Egap_Ez_imag_V_per_m",
        "Eparallel_real", "Eparallel_imag", "Eparallel_abs", "Eparallel_phase_deg",
        "Egap_total_abs", "Egap_total_over_E0", "lmax_used",
        "magnitude_convergence_percent", "phase_convergence_deg", "convergence_status"]
    field_df[usable_columns].rename(columns={
        "Eparallel_real":"Egap_parallel_real_V_per_m", "Eparallel_imag":"Egap_parallel_imag_V_per_m",
        "Eparallel_abs":"Egap_parallel_abs_V_per_m", "Eparallel_phase_deg":"Egap_parallel_phase_deg",
        "Egap_total_abs":"Egap_total_abs_V_per_m"}).to_csv(output / "USABLE_EGAP_TABLE.csv", index=False)
    plot_results(field_df, history_df, output)
    return field_df, history_df


def plot_results(fields, history, output):
    fig, ax = plt.subplots()
    for angle, group in fields.groupby("polarization_deg"):
        ax.plot(group.gap_nm, group.Egap_total_over_E0, "o-", label=f"{angle:g}°")
    ax.set(xlabel="gap (nm)", ylabel=r"$|E_{gap}/E_0|$"); ax.legend(title="polarization")
    fig.tight_layout(); fig.savefig(output / "Egap_MIEPY_vs_gap.svg"); plt.close(fig)
    fig, ax = plt.subplots()
    selected = history[history.gap_nm.isin([1, 2, 3, 6]) & (history.polarization_deg == 0)]
    for gap, group in selected.groupby("gap_nm"):
        ax.plot(group.lmax, group.Egap_abs, "o-", label=f"g={gap:g} nm")
    ax.set(xlabel="lmax", ylabel=r"$|E_{gap,x}|$ (V/m)"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "multipole_convergence_selected_gaps.svg"); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("results/level3B"))
    run(parser.parse_args().output)
