"""Level 2 representative simulation and reproducible output writer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np

from .coupled_dipole import solve_coupled_dipoles
from .excitation import gaussian_incident_field
from .geometry import GeometryConfig, generate_aggregate, minimum_surface_gap
from .polarizability import sphere_polarizability


def run_level2(output_dir: str | Path = "results/level2", seed: int = 20260813,
               n_particles: int = 200) -> dict[str, object]:
    """Run one Level 2 realization; no propagation coefficient is fitted."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = generate_aggregate(GeometryConfig(n_particles=n_particles, seed=seed))
    positions_3d_m = np.column_stack((aggregate.positions_nm, np.zeros(n_particles))) * 1e-9

    # Explicit representative inputs, not a claim of experimental validation.
    wavelength_nm = 633.0
    medium_epsilon = 1.0 + 0j
    silver_epsilon = -15.0 + 1.0j
    alpha = sphere_polarizability(aggregate.radii_nm * 1e-9, wavelength_nm * 1e-9,
                                  silver_epsilon, medium_epsilon)
    source_index = int(np.argmin(aggregate.positions_nm[:, 0]))
    source_nm = aggregate.positions_nm[source_index].copy()
    source_3d_m = np.array([source_nm[0], source_nm[1], 0.0]) * 1e-9
    source_waist_nm = 65.0
    incident = gaussian_incident_field(positions_3d_m, source_3d_m,
                                       source_waist_nm * 1e-9)
    wave_number = 2 * np.pi * np.sqrt(medium_epsilon) / (wavelength_nm * 1e-9)
    solution = solve_coupled_dipoles(positions_3d_m, alpha, incident, wave_number,
                                     medium_epsilon)
    intensities = np.sum(np.abs(solution.dipoles_c_m) ** 2, axis=1)

    with (output / "particles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["particle_id", "x_nm", "y_nm", "radius_nm", "degree",
                         "incident_field_magnitude_v_per_m", "dipole_intensity_c2_m2"])
        for i in range(n_particles):
            writer.writerow([i, *aggregate.positions_nm[i], aggregate.radii_nm[i],
                             aggregate.degrees[i], np.linalg.norm(incident[i]), intensities[i]])
    with (output / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["particle_i", "particle_j", "surface_gap_nm"])
        for edge, gap in zip(aggregate.edges, aggregate.edge_gaps_nm):
            writer.writerow([*edge, gap])

    _plot_geometry(output / "aggregate_geometry.svg", aggregate)
    _plot_graph(output / "connectivity_graph.svg", aggregate)
    _plot_intensity(output / "dipole_intensity_map.svg", aggregate, intensities, source_nm)
    for svg_path in output.glob("*.svg"):
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8",
        )

    unique, counts = np.unique(aggregate.degrees, return_counts=True)
    summary = {
        "N": n_particles,
        "seed": seed,
        "radius_mean_nm": float(aggregate.radii_nm.mean()),
        "radius_std_nm": float(aggregate.radii_nm.std()),
        "gap_mean_nm": float(aggregate.edge_gaps_nm.mean()),
        "gap_std_nm": float(aggregate.edge_gaps_nm.std()),
        "coordination_number_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
        "number_of_connected_components": 1,
        "mean_degree": float(aggregate.degrees.mean()),
        "maximum_degree": int(aggregate.degrees.max()),
        "source_location_nm": source_nm.tolist(),
        "source_waist_nm": source_waist_nm,
        "minimum_surface_gap_nm": minimum_surface_gap(aggregate),
        "relative_linear_residual": solution.residual_relative,
        "wavelength_nm": wavelength_nm,
        "silver_relative_permittivity_assumption": str(silver_epsilon),
    }
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows((key, value) for key, value in summary.items())
    return summary


def _segments(aggregate):
    return np.asarray([[aggregate.positions_nm[i], aggregate.positions_nm[j]]
                       for i, j in aggregate.edges])


def _setup_axis():
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.set_aspect("equal")
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("y (nm)")
    return figure, axis


def _plot_geometry(path, aggregate):
    figure, axis = _setup_axis()
    for position, radius in zip(aggregate.positions_nm, aggregate.radii_nm):
        axis.add_patch(plt.Circle(position, radius, facecolor="#aeb7c2", edgecolor="#34495e", lw=.35))
    axis.autoscale_view()
    axis.set_title("Stochastic silver nanoparticle aggregate")
    figure.tight_layout(); figure.savefig(path, format="svg"); plt.close(figure)


def _plot_graph(path, aggregate):
    figure, axis = _setup_axis()
    axis.add_collection(LineCollection(_segments(aggregate), colors="#9aa0a6", linewidths=.6))
    scatter = axis.scatter(*aggregate.positions_nm.T, c=aggregate.degrees, s=12, cmap="viridis")
    figure.colorbar(scatter, ax=axis, label="coordination number")
    axis.autoscale_view(); axis.set_title("Near-contact connectivity graph")
    figure.tight_layout(); figure.savefig(path, format="svg"); plt.close(figure)


def _plot_intensity(path, aggregate, intensities, source_nm):
    figure, axis = _setup_axis()
    positive = np.maximum(intensities, np.finfo(float).tiny)
    scatter = axis.scatter(*aggregate.positions_nm.T, c=positive, s=24, cmap="magma",
                           norm=LogNorm(vmin=positive.min(), vmax=positive.max()))
    axis.scatter(*source_nm, marker="*", s=100, color="cyan", edgecolor="black", label="Gaussian center")
    axis.legend(); figure.colorbar(scatter, ax=axis, label=r"$|p|^2$ (C$^2$ m$^2$)")
    axis.set_title("Complex coupled-dipole intensity map")
    figure.tight_layout(); figure.savefig(path, format="svg"); plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/level2")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--n-particles", type=int, default=200)
    arguments = parser.parse_args()
    for key, value in run_level2(arguments.output_dir, arguments.seed, arguments.n_particles).items():
        print(f"{key}: {value}")
