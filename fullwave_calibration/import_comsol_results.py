"""Validate and import results produced by an actual COMSOL dimer run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import CONFIG

EXPECTED_EPSILON = complex(-18.320291283372367, 0.4792932084309133)
REQUIRED = {"gap_nm", "radius_nm", "wavelength_nm", "polarization_deg",
    "epsilon_Ag_real", "epsilon_Ag_imag", "E0_V_per_m", "convergence_status",
    "Ex_real", "Ex_imag", "Ey_real", "Ey_imag", "Ez_real", "Ez_imag"}


def validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"missing COMSOL columns: {sorted(missing)}")
    checks = {
        "radius": np.allclose(frame.radius_nm, CONFIG.radius_nm),
        "wavelength": np.allclose(frame.wavelength_nm, CONFIG.wavelength_nm),
        "gap": frame.gap_nm.isin(CONFIG.gaps_nm).all(),
        "polarization": frame.polarization_deg.isin(CONFIG.polarizations_deg).all(),
        "epsilon": np.allclose(frame.epsilon_Ag_real, EXPECTED_EPSILON.real) and
                   np.allclose(frame.epsilon_Ag_imag, EXPECTED_EPSILON.imag),
        "E0": np.allclose(frame.E0_V_per_m, CONFIG.e0_v_per_m),
        "convergence": frame.convergence_status.eq("PASS").all(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("incompatible COMSOL results: " + ", ".join(failed))


def main(path: Path) -> None:
    comsol = pd.read_csv(path); validate(comsol)
    miepy = pd.read_csv("results/level3B/data/Egap_MIEPY_dimer.csv")
    merged = comsol.merge(miepy, on=["gap_nm", "radius_nm", "wavelength_nm", "polarization_deg"],
                          suffixes=("_COMSOL", "_MIEPY"), validate="one_to_one")
    output = Path("results/level3B"); merged.to_csv(output / "COMSOL_vs_MIEPY_comparison.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cabs = np.sqrt(sum((merged[f"{v}_real"]**2 + merged[f"{v}_imag"]**2) for v in ("Ex", "Ey", "Ez")))
    ax.scatter(merged.Egap_total_abs, cabs); ax.set(xlabel="MiePy |Egap|", ylabel="COMSOL |Egap|")
    fig.tight_layout(); fig.savefig(output / "COMSOL_vs_MIEPY_vs_PDA.svg"); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("csv", type=Path); main(parser.parse_args().csv)
