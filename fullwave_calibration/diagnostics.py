"""Diagnose immutable Level-3B inputs and verify required outputs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "Level-3A.5 frozen network directory": ROOT / "results/level3A_5",
    "Frozen complex hotspot fields": ROOT / "results/level3A/data/hotspots.csv",
    "Frozen particle coordinates": ROOT / "results/level2_6/data/particles.csv",
    "Level-3A.5 PDA implementation": ROOT / "stochastic_2d",
    "USABLE_EGAP_TABLE.csv": ROOT / "results/level3B/USABLE_EGAP_TABLE.csv",
    "Egap_MIEPY_dimer.csv": ROOT / "results/level3B/data/Egap_MIEPY_dimer.csv",
}
REQUIRED_OUTPUTS = tuple(ROOT / path for path in (
    "results/level3B/data/Egap_PDA_dimer.csv",
    "results/level3B/data/gap_calibration_table.csv",
    "results/level3B/PDA_vs_MIEPY_longitudinal.svg",
    "results/level3B/Cparallel_magnitude_vs_gap.svg",
    "results/level3B/Cparallel_phase_vs_gap.svg",
    "results/level3B/Cperp_magnitude_vs_gap.svg",
    "results/level3B/Cperp_phase_vs_gap.svg",
    "results/level3B/component_correction_validation.svg",
    "results/level3B/data/component_correction_validation.csv",
    "results/level3B/network_M4_PDA.svg",
    "results/level3B/network_M4_fullwave_corrected.svg",
    "results/level3B/remote_hotspot_ranking_before_after.svg",
    "results/level3B/strongest_corrected_remote_hotspot.svg",
    "results/level3B/data/network_hotspots_fullwave_corrected.csv",
    "results/level3B/M4_PDA_vs_corrected_scatter.svg",
    "results/level3B/downstream_enhancement_before_after.svg",
))


def exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> int:
    print(f"{'INPUT':43} STATUS")
    for name, path in INPUTS.items():
        found = path.exists() and (not path.is_file() or path.stat().st_size > 0)
        print(f"{name:43} {'FOUND' if found else 'MISSING':7} {path.relative_to(ROOT)}")
    missing = [path for path in REQUIRED_OUTPUTS if not exists_nonempty(path)]
    print("\n" + ("LEVEL 3B COMPLETE" if not missing else "LEVEL 3B INCOMPLETE"))
    for path in missing:
        print(f"MISSING {path.relative_to(ROOT)}")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
