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

from .coupled_dipole import scattered_field, solve_coupled_dipoles
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
    scattered = scattered_field(
        positions_3d_m, solution.dipoles_c_m, wave_number, medium_epsilon
    )
    total = incident + scattered
    intensities = np.sum(np.abs(solution.dipoles_c_m) ** 2, axis=1)
    normalized = intensities / intensities.max()
    incident_intensity = np.sum(np.abs(incident) ** 2, axis=1)
    scattered_intensity = np.sum(np.abs(scattered) ** 2, axis=1)
    total_intensity = np.sum(np.abs(total) ** 2, axis=1)
    illumination_fraction = incident_intensity / incident_intensity.max()
    illuminated = illumination_fraction >= 1e-4
    remote = ~illuminated
    remote_indices = np.flatnonzero(remote)
    if not len(remote_indices):
        raise RuntimeError("Gaussian threshold produced no remote particles")
    strongest_remote = int(remote_indices[np.argmax(intensities[remote])])
    path = _shortest_path(aggregate.edges, source_index, strongest_remote, n_particles)
    graph_distance = _graph_distances(aggregate.edges, source_index, n_particles)
    downstream_pairs = [
        (int(i), int(j)) for i, j in aggregate.edges
        if graph_distance[int(j)] == graph_distance[int(i)] + 1 and intensities[int(j)] > intensities[int(i)]
    ] + [
        (int(j), int(i)) for i, j in aggregate.edges
        if graph_distance[int(i)] == graph_distance[int(j)] + 1 and intensities[int(i)] > intensities[int(j)]
    ]

    with (output / "particles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["particle_id", "x_nm", "y_nm", "radius_nm", "degree",
                         "incident_intensity_v2_per_m2", "scattered_intensity_v2_per_m2",
                         "total_intensity_v2_per_m2", "dipole_intensity_c2_m2",
                         "normalized_dipole_intensity", "directly_illuminated", "remote"])
        for i in range(n_particles):
            writer.writerow([i, *aggregate.positions_nm[i], aggregate.radii_nm[i],
                             aggregate.degrees[i], incident_intensity[i], scattered_intensity[i],
                             total_intensity[i], intensities[i], normalized[i],
                             bool(illuminated[i]), bool(remote[i])])
    with (output / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["particle_i", "particle_j", "surface_gap_nm"])
        for edge, gap in zip(aggregate.edges, aggregate.edge_gaps_nm):
            writer.writerow([*edge, gap])

    _plot_geometry(output / "aggregate_geometry.svg", aggregate)
    _plot_graph(output / "connectivity_graph.svg", aggregate)
    _plot_intensity(output / "dipole_intensity_map.svg", aggregate, intensities, source_nm)
    _plot_distributions(output, aggregate)
    _plot_field_maps(output, aggregate, incident_intensity, scattered_intensity,
                     total_intensity)
    _plot_normalized_dipoles(output / "normalized_dipole_intensity.svg", aggregate,
                             normalized, source_nm)
    _plot_path(output / "geodesic_path.svg", aggregate, path, normalized)
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
        "gap_median_nm": float(np.median(aggregate.edge_gaps_nm)),
        "gap_std_nm": float(aggregate.edge_gaps_nm.std()),
        "gap_min_nm": float(aggregate.edge_gaps_nm.min()),
        "gap_max_nm": float(aggregate.edge_gaps_nm.max()),
        "coordination_number_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
        "number_of_connected_components": 1,
        "mean_degree": float(aggregate.degrees.mean()),
        "maximum_degree": int(aggregate.degrees.max()),
        "source_location_nm": source_nm.tolist(),
        "source_waist_nm": source_waist_nm,
        "polarization_xyz": "[1.0, 0.0, 0.0]",
        "incident_field_peak_amplitude_v_per_m": 1.0,
        "surrounding_refractive_index": 1.0,
        "minimum_surface_gap_nm": minimum_surface_gap(aggregate),
        "relative_linear_residual": solution.residual_relative,
        "wavelength_nm": wavelength_nm,
        "silver_relative_permittivity_assumption": str(silver_epsilon),
        "silver_dielectric_source": "UNSOURCED ILLUSTRATIVE ASSUMPTION; not physical validation",
        "directly_illuminated_particle_count": int(illuminated.sum()),
        "remote_particle_count": int(remote.sum()),
        "remote_threshold_incident_intensity_fraction": 1e-4,
        "finite_remote_dipole_excitation": bool(np.any(np.isfinite(intensities[remote]) & (intensities[remote] > 0))),
        "strongest_remote_particle_id": strongest_remote,
        "strongest_remote_normalized_dipole_intensity": float(normalized[strongest_remote]),
        "geodesic_path_edge_count": len(path) - 1,
        "downstream_increase_on_path": bool(np.any(np.diff(intensities[path]) > 0)),
        "any_natural_downstream_increase": bool(downstream_pairs),
        "example_upstream_particle_id": downstream_pairs[0][0] if downstream_pairs else "none",
        "example_downstream_particle_id": downstream_pairs[0][1] if downstream_pairs else "none",
    }
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows((key, value) for key, value in summary.items())
    _write_validation_csvs(output, aggregate, path, incident, scattered, total,
                           intensities, normalized, illumination_fraction)
    return summary


def _shortest_path(edges, start, target, n_particles):
    adjacency = [[] for _ in range(n_particles)]
    for i, j in edges:
        adjacency[int(i)].append(int(j)); adjacency[int(j)].append(int(i))
    parents = {start: None}; queue = [start]
    for node in queue:
        if node == target:
            break
        for neighbor in sorted(adjacency[node]):
            if neighbor not in parents:
                parents[neighbor] = node; queue.append(neighbor)
    path = [target]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return np.asarray(path[::-1], dtype=int)


def _graph_distances(edges, start, n_particles):
    adjacency = [[] for _ in range(n_particles)]
    for i, j in edges:
        adjacency[int(i)].append(int(j)); adjacency[int(j)].append(int(i))
    distances = np.full(n_particles, -1, dtype=int); distances[start] = 0
    queue = [start]
    for node in queue:
        for neighbor in adjacency[node]:
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1; queue.append(neighbor)
    return distances


def _write_validation_csvs(output, aggregate, path, incident, scattered, total,
                           intensities, normalized, illumination_fraction):
    with (output / "fields.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["particle_id", *[f"E_inc_{x}" for x in ("x_real", "x_imag", "y_real", "y_imag", "z_real", "z_imag")],
                         *[f"E_scat_{x}" for x in ("x_real", "x_imag", "y_real", "y_imag", "z_real", "z_imag")],
                         *[f"E_total_{x}" for x in ("x_real", "x_imag", "y_real", "y_imag", "z_real", "z_imag")]])
        for i in range(len(incident)):
            components = [value for field in (incident[i], scattered[i], total[i])
                          for component in field for value in (component.real, component.imag)]
            writer.writerow([i, *components])
    with (output / "geodesic_path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["path_step", "particle_id", "normalized_dipole_intensity",
                         "incident_intensity_fraction", "increase_from_previous"])
        for step, particle in enumerate(path):
            increase = "" if step == 0 else bool(intensities[particle] > intensities[path[step-1]])
            writer.writerow([step, particle, normalized[particle], illumination_fraction[particle], increase])
    for filename, values, label in (
        ("radius_distribution.csv", aggregate.radii_nm, "radius_nm"),
        ("gap_distribution.csv", aggregate.edge_gaps_nm, "surface_gap_nm"),
        ("coordination_distribution.csv", aggregate.degrees, "coordination_number"),
    ):
        unique, counts = np.unique(values, return_counts=True) if label == "coordination_number" else np.histogram(values, bins="auto")[:2]
        if label != "coordination_number":
            counts, unique = unique, counts
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([label if label == "coordination_number" else "bin_left", "bin_right", "count", "probability"])
            if label == "coordination_number":
                for value, count in zip(unique, counts): writer.writerow([value, value, count, count / len(values)])
            else:
                for left, right, count in zip(unique[:-1], unique[1:], counts): writer.writerow([left, right, count, count / len(values)])


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


def _plot_distributions(output, aggregate):
    specifications = (
        (aggregate.radii_nm, "Particle radius (nm)", "radius_distribution.svg"),
        (aggregate.edge_gaps_nm, "Surface gap (nm)", "gap_distribution.svg"),
    )
    for values, xlabel, filename in specifications:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(values, bins="auto", density=True, color="#4878a8", edgecolor="white")
        axis.set(xlabel=xlabel, ylabel="Probability density", title=f"{xlabel} distribution")
        figure.tight_layout(); figure.savefig(output / filename, format="svg"); plt.close(figure)
    unique, counts = np.unique(aggregate.degrees, return_counts=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(unique, counts / counts.sum(), color="#54a24b")
    axis.set(xlabel="Coordination number z", ylabel="P(z)", title="Coordination distribution")
    axis.set_xticks(unique)
    figure.tight_layout(); figure.savefig(output / "coordination_distribution.svg", format="svg"); plt.close(figure)


def _plot_field_maps(output, aggregate, incident, scattered, total):
    for values, symbol, filename in (
        (incident, r"$|E_{inc}|^2$", "incident_field_intensity.svg"),
        (scattered, r"$|E_{scat}|^2$", "scattered_field_intensity.svg"),
        (total, r"$|E_{total}|^2$", "total_field_intensity.svg"),
    ):
        figure, axis = _setup_axis()
        positive = np.maximum(values, np.finfo(float).tiny)
        scatter = axis.scatter(*aggregate.positions_nm.T, c=positive, s=25, cmap="viridis",
                               norm=LogNorm(vmin=positive.min(), vmax=positive.max()))
        figure.colorbar(scatter, ax=axis, label=symbol + r" (V$^2$/m$^2$)")
        axis.set_title(symbol + " at particle centers")
        figure.tight_layout(); figure.savefig(output / filename, format="svg"); plt.close(figure)


def _plot_normalized_dipoles(path, aggregate, normalized, source_nm):
    figure, axis = _setup_axis()
    log_values = np.log10(np.maximum(normalized, np.finfo(float).tiny))
    scatter = axis.scatter(*aggregate.positions_nm.T, c=log_values, s=25, cmap="magma")
    axis.scatter(*source_nm, marker="*", s=100, color="cyan", edgecolor="black")
    figure.colorbar(scatter, ax=axis, label=r"$\log_{10}(|p_i|^2/\max_j|p_j|^2)$")
    axis.set_title("Normalized complex-dipole excitation")
    figure.tight_layout(); figure.savefig(path, format="svg"); plt.close(figure)


def _plot_path(pathname, aggregate, path, normalized):
    figure, (network_axis, profile_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    network_axis.set_aspect("equal")
    network_axis.add_collection(LineCollection(_segments(aggregate), colors="#dddddd", linewidths=.4))
    network_axis.plot(*aggregate.positions_nm[path].T, "o-", color="#e45756", markersize=3)
    network_axis.autoscale_view(); network_axis.set_title("Source-to-remote geodesic path")
    network_axis.set(xlabel="x (nm)", ylabel="y (nm)")
    profile_axis.semilogy(np.arange(len(path)), normalized[path], "o-")
    profile_axis.set(xlabel="Geodesic path step", ylabel=r"$|p_i|^2/\max_j|p_j|^2$",
                     title="Excitation along natural path")
    profile_axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(pathname, format="svg"); plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/level2")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--n-particles", type=int, default=200)
    arguments = parser.parse_args()
    for key, value in run_level2(arguments.output_dir, arguments.seed, arguments.n_particles).items():
        print(f"{key}: {value}")
