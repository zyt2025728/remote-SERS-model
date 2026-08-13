#!/usr/bin/env python3
"""Run a reproducible two-dimensional Ag nanoparticle coupled-dipole model.

The particles lie in the x-y plane and are illuminated with z-polarized light.
Consequently each particle has one (z) dipole degree of freedom, while the
retarded free-space Green function and the full multiple-scattering solve are
retained.  SI dipole moments are reported in the output table.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import linregress


WAVELENGTH_NM = 633.0
RADIUS_NM = 18.0
N_ROWS, N_COLS = 7, 8
PITCH_NM = 45.0
MEDIUM_INDEX = 1.0
SILVER_EPSILON = -15.8 + 1.05j
E0_V_PER_M = 1.0
SEED = 20260813


def make_network() -> np.ndarray:
    """Return a weakly disordered, staggered 2-D particle network in metres."""
    rng = np.random.default_rng(SEED)
    points = []
    dy = PITCH_NM * np.sqrt(3) / 2
    for col in range(N_COLS):
        for row in range(N_ROWS):
            x = col * PITCH_NM
            y = (row + 0.5 * (col % 2)) * dy
            points.append((x, y))
    points = np.asarray(points, dtype=float)
    points += rng.normal(0.0, 0.65, points.shape)
    points[:, 0] -= points[:, 0].min()
    points[:, 1] -= points[:, 1].mean()
    return points * 1e-9


def transverse_green(r: float, k: float) -> complex:
    """z-z electric Green tensor element for separation in the x-y plane."""
    return np.exp(1j * k * r) / (4 * np.pi) * (
        k**2 / r - (1 / r**3 - 1j * k / r**2)
    )


def run_simulation() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    positions = make_network()
    n_particles = len(positions)
    wavelength = WAVELENGTH_NM * 1e-9
    radius = RADIUS_NM * 1e-9
    k = 2 * np.pi * MEDIUM_INDEX / wavelength
    eps0 = 8.8541878128e-12

    # Radiation-corrected Clausius-Mossotti polarizability, alpha/eps0 [m^3].
    alpha0 = 4 * np.pi * radius**3 * (SILVER_EPSILON - 1) / (SILVER_EPSILON + 2)
    alpha = alpha0 / (1 - 1j * k**3 * alpha0 / (6 * np.pi))

    # A localized excitation at the left edge launches energy into the network.
    excitation_width = 0.55 * PITCH_NM * 1e-9
    incident = E0_V_PER_M * np.exp(-(positions[:, 0] / excitation_width) ** 2)
    matrix = np.eye(n_particles, dtype=complex)
    for i in range(n_particles):
        for j in range(n_particles):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                matrix[i, j] = -alpha * transverse_green(distance, k)

    local_field = solve(matrix, incident, assume_a="gen")
    dipole = eps0 * alpha * local_field
    intensity = np.abs(local_field / E0_V_PER_M) ** 2
    distance_nm = positions[:, 0] * 1e9

    # Average particles in each nominal column to suppress coherent oscillations.
    columns = np.repeat(np.arange(N_COLS), N_ROWS)
    propagation = pd.DataFrame(
        {"column": columns, "distance_nm": distance_nm, "normalized_intensity": intensity}
    ).groupby("column", as_index=False).agg(
        distance_nm=("distance_nm", "mean"),
        normalized_intensity=("normalized_intensity", "mean"),
        intensity_std=("normalized_intensity", "std"),
    )
    propagation["relative_intensity"] = (
        propagation.normalized_intensity / propagation.normalized_intensity.iloc[0]
    )
    propagation["ln_relative_intensity"] = np.log(propagation.relative_intensity)

    # Fit beyond the launch column.  I/I(0)=exp(-mu*x), L=1/mu.
    fit_data = propagation.iloc[1:]
    fit = linregress(fit_data.distance_nm, fit_data.ln_relative_intensity)
    attenuation = -fit.slope
    propagation["fitted_relative_intensity"] = np.exp(fit.intercept + fit.slope * propagation.distance_nm)
    propagation["fitted_ln_relative_intensity"] = fit.intercept + fit.slope * propagation.distance_nm

    particle = pd.DataFrame(
        {
            "particle_id": np.arange(n_particles),
            "x_nm": distance_nm,
            "y_nm": positions[:, 1] * 1e9,
            "dipole_real_Cm": dipole.real,
            "dipole_imag_Cm": dipole.imag,
            "local_field_real_Vm": local_field.real,
            "local_field_imag_Vm": local_field.imag,
            "field_magnitude_Vm": np.abs(local_field),
            "normalized_E2": intensity,
            "hotspot_intensity": intensity,
            "propagation_distance_nm": distance_nm,
        }
    )
    nearest = []
    for i, point in enumerate(positions):
        nearest.append(np.min(np.linalg.norm(positions[np.arange(n_particles) != i] - point, axis=1)))
    mean_center_nm = np.mean(nearest) * 1e9
    results = {
        "wavelength_nm": WAVELENGTH_NM,
        "radius_nm": RADIUS_NM,
        "number_of_particles": n_particles,
        "mean_center_distance_nm": mean_center_nm,
        "mean_gap_nm": mean_center_nm - 2 * RADIUS_NM,
        "maximum_normalized_hotspot_intensity": intensity.max(),
        "attenuation_coefficient_per_nm": attenuation,
        "fit_r_squared": fit.rvalue**2,
        "propagation_length_nm": 1 / attenuation if attenuation > 0 else np.nan,
        "matrix_condition_number": np.linalg.cond(matrix),
    }
    return particle, propagation, results


def save_figures(particle: pd.DataFrame, propagation: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(particle.x_nm, particle.y_nm, s=130, facecolor="silver", edgecolor="black")
    ax.axvspan(-10, 30, color="gold", alpha=0.2, label="excitation region")
    ax.set(xlabel="x (nm)", ylabel="y (nm)", title="2D Ag nanoparticle network")
    ax.axis("equal"); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "network_geometry.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    values = particle.hotspot_intensity / particle.hotspot_intensity.max()
    sc = ax.scatter(particle.x_nm, particle.y_nm, c=values, s=155, cmap="inferno", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label=r"Normalized $|E|^2$ (relative to maximum)")
    ax.set(xlabel="x (nm)", ylabel="y (nm)", title="Hotspot intensity map"); ax.axis("equal"); fig.tight_layout()
    fig.savefig(figure_dir / "hotspot_map.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(propagation.distance_nm, propagation.relative_intensity,
                yerr=propagation.intensity_std / propagation.normalized_intensity.iloc[0], marker="o", capsize=3)
    ax.set(xlabel="Propagation distance (nm)", ylabel="Mean intensity / launch intensity", title="Hotspot propagation")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir / "intensity_vs_distance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(propagation.distance_nm, propagation.ln_relative_intensity, label="simulation")
    ax.plot(propagation.distance_nm, propagation.fitted_ln_relative_intensity, label="linear attenuation fit")
    ax.set(xlabel="Propagation distance (nm)", ylabel="ln(relative intensity)", title="Effective attenuation fit")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "attenuation_fit.png"); plt.close(fig)


def main() -> None:
    data_dir, figure_dir = Path("results/data"), Path("results/figures")
    data_dir.mkdir(parents=True, exist_ok=True)
    particle, propagation, results = run_simulation()
    if not np.isfinite(particle.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("Simulation produced NaN or Inf")
    particle.to_csv(data_dir / "particle_results.csv", index=False)
    propagation.to_csv(data_dir / "propagation_results.csv", index=False)
    pd.Series(results, name="value").rename_axis("parameter").to_csv(data_dir / "simulation_summary.csv")
    save_figures(particle, propagation, figure_dir)
    print("2D Ag coupled-dipole simulation completed")
    for key, value in results.items():
        print(f"{key}: {value:.8g}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
