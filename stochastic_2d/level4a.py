"""Level 4A stochastic ensemble statistics for the calibrated 2D CDA model.

This module deliberately imports the frozen Level-3B calibration as data.  It
does not import or execute the MiePy production sweep.  ``run_pilot`` changes
only ``GeometryConfig.seed`` between independently reproducible realizations.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .coupled_dipole import solve_coupled_dipoles
from .excitation import gaussian_incident_field
from .geometry import GeometryConfig, generate_aggregate
from .hotspots import (classify_hotspots, construct_hotspots, fields_at_hotspots,
                       hotspot_distances)
from .polarizability import sphere_polarizability

PILOT_SEEDS = tuple(20260813 + 104729 * i for i in range(10))
SUPPORTED_ENSEMBLE_SIZES = (10, 20, 30, 40, 50, 75, 100)
CALIBRATION_PATH = Path("results/level3B/data/gap_calibration_table.csv")
STATUS_PATH = Path("results/level3B/data/solver_status.csv")
GEODESIC_BIN_EDGES_NM = np.arange(0.0, 2100.0, 100.0)
QUANTILES = (.05, .25, .50, .75, .95)

# These values are the arguments and defaults used by validation_level3a_5.run.
BASE_GEOMETRY = GeometryConfig(n_particles=200, seed=20260813)
MODEL = {
    "wavelength_nm": 633.0,
    "nanoparticle_material": "Ag sphere",
    "particle_relative_permittivity": {"real": -15.0, "imag": 1.0},
    "surrounding_medium": "homogeneous vacuum, relative permittivity 1+0j",
    "source_particle_rule": "particle with minimum x coordinate",
    "aggregate_generation": "validated connected stochastic growth in stochastic_2d.geometry.generate_aggregate",
    "branching_parameters": "four of every five children favor low-degree parents; every fifth favors compact centroid growth; angular disorder from geometry.positional_disorder_rad",
    "coordination_constraints": "no hard coordination cap; overlap/minimum-gap rejection and inverse-(1+degree) branch weighting",
    "minimum_separation_rule": "all particle pairs have surface separation >= geometry.gap_min_nm",
    "particle_radius_distribution": "frozen truncated normal specified by geometry radius fields",
    "surface_gap_distribution": "frozen truncated normal specified by geometry gap fields",
    "gaussian_beam_waist_nm": 55.0,
    "incident_field_amplitude_V_per_m": {"real": 1.0, "imag": 0.0},
    "incident_polarization_xyz": [1.0, 0.0, 0.0],
    "incident_propagation_convention": "localized scalar Gaussian envelope; exp(-i omega t)",
    "hotspot_definition": "midpoint between facing surfaces of every 1-6 nm near contact",
    "near_contact_graph_criterion_nm": [1.0, 6.0],
    "source_region_definition": "M2_inc >= 1e-6",
    "transition_region_definition": "not source and not remote",
    "remote_region_definition": "M2_inc < 1e-6 and Euclidean source distance > 500 nm",
    "plasmon_remote_definition": "remote and M2_scat/M2_inc >= 100",
    "geodesic_distance_definition": "shortest hotspot-adjacency path weighted by midpoint distance",
    "geodesic_launch_definition": "strongest CDA-M4 source hotspot",
    "downstream_direction_definition": "strictly increasing corrected weighted geodesic distance",
    "CDA_polarizability": "Clausius-Mossotti sphere polarizability",
    "radiation_correction": "alpha=alpha_static/(1-i*k^3*alpha_static/(6*pi*epsilon0*epsilon_m))",
    "retarded_dyadic_Green_tensor": "full 3D electric dyadic retaining r^-3, r^-2, r^-1 terms",
    "CDA_solver": "dense complex self-consistent solve using all ordered particle pairs",
    "fullwave_calibration_table": str(CALIBRATION_PATH),
    "C_parallel": "piecewise-linear interpolation of complex real/imaginary columns; no extrapolation",
    "C_perp": "piecewise-linear interpolation of complex real/imaginary columns; no extrapolation",
    "calibration_domain_nm": [1.0, 6.0],
    "solver_backend": "miepy_GMMT",
    "COMSOL_numerical_status": "NOT EXECUTED",
    "geodesic_bin_edges_nm": GEODESIC_BIN_EDGES_NM.tolist(),
}


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(type(value).__name__)


def frozen_configuration() -> dict:
    """Return the complete immutable model configuration (seed excluded)."""
    geometry = asdict(BASE_GEOMETRY)
    geometry.pop("seed")
    return {"model_name": "2D stochastic retarded coupled-dipole approximation (CDA) model",
            "only_variable_parameter": "random_seed", "geometry": geometry, **MODEL}


def frozen_configuration_markdown(configuration=None) -> str:
    """Render the frozen machine configuration as a human-auditable record."""
    configuration = configuration or frozen_configuration()
    lines = ["# Frozen Level 4A configuration", "",
             "The network is a **2D stochastic retarded coupled-dipole approximation (CDA) model** with full-wave MiePy/GMMT isolated-gap calibration.", "",
             "The **only** realization-dependent parameter is `random_seed`. All values below are identical for every pilot realization.", "",
             "## Geometry and stochastic generator", ""]
    for key, value in configuration["geometry"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Electromagnetic, source, region, and analysis definitions", ""])
    for key, value in configuration.items():
        if key not in {"geometry", "model_name", "only_variable_parameter"}:
            rendered = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            lines.append(f"- **{key.replace('_', ' ')}:** `{rendered}`")
    lines.extend(["", "Corrected M4 is a **full-wave-calibrated electromagnetic SERS proxy**, not experimental Raman intensity.", ""])
    return "\n".join(lines)


def verify_level3b() -> None:
    """Fail before simulation if any frozen prerequisite or provenance is wrong."""
    required = [Path("stochastic_2d"), Path("results/level3A_5"), CALIBRATION_PATH,
                Path("results/level3B/data/network_hotspots_fullwave_corrected.csv"),
                Path("results/level3B/USABLE_EGAP_TABLE.csv"), Path("fullwave_calibration")]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing frozen Level-3B prerequisites: " + ", ".join(missing))
    calibration = pd.read_csv(CALIBRATION_PATH)
    columns = {"Cparallel_real", "Cparallel_imag", "Cperp_real", "Cperp_imag"}
    if not columns <= set(calibration.columns):
        raise ValueError("Level-3B complex calibration columns are incomplete")
    with STATUS_PATH.open(newline="") as stream:
        status = {row[0]: ",".join(row[1:]) for row in list(csv.reader(stream))[1:]}
    if status["solver_backend"] != "miepy_GMMT" or status["COMSOL_numerical_status"] != "NOT EXECUTED":
        raise ValueError("unexpected frozen Level-3B solver provenance")


class ComplexGapCalibration:
    """Bounds-checked piecewise-linear complex Level-3B correction."""
    def __init__(self, path: Path = CALIBRATION_PATH):
        table = pd.read_csv(path).sort_values("gap_nm")
        self.gaps = table.gap_nm.to_numpy(float)
        self.parallel = table.Cparallel_real.to_numpy() + 1j * table.Cparallel_imag.to_numpy()
        self.perp = table.Cperp_real.to_numpy() + 1j * table.Cperp_imag.to_numpy()

    def interpolate(self, gap_nm):
        values = np.asarray(gap_nm, float)
        valid = (values >= self.gaps[0]) & (values <= self.gaps[-1])
        cp = np.full(values.shape, np.nan + 1j*np.nan, complex)
        ct = np.full(values.shape, np.nan + 1j*np.nan, complex)
        cp[valid] = (np.interp(values[valid], self.gaps, self.parallel.real)
                     + 1j*np.interp(values[valid], self.gaps, self.parallel.imag))
        ct[valid] = (np.interp(values[valid], self.gaps, self.perp.real)
                     + 1j*np.interp(values[valid], self.gaps, self.perp.imag))
        return cp, ct, valid


def directed_transitions(adjacency, distance):
    """Return unique adjacency transitions oriented toward larger geodesic distance."""
    result = []
    for u, neighbours in enumerate(adjacency):
        for v in neighbours:
            if u >= v or distance[u] == distance[v]:
                continue
            result.append((u, v) if distance[u] < distance[v] else (v, u))
    return result


def gamma_de(upstream, downstream):
    """Downstream enhancement, with particle-radius symbol R kept unambiguous."""
    return np.asarray(downstream) / np.asarray(upstream)


def equal_seed_mean(frame: pd.DataFrame, value: str) -> float:
    """Mean giving each seed equal weight irrespective of hotspot count."""
    return float(frame.groupby("seed")[value].mean().mean())


def _save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _field_columns(prefix, field):
    return {f"{prefix}_{axis}_{part}": field[:, i].real if part == "real" else field[:, i].imag
            for i, axis in enumerate("xyz") for part in ("real", "imag")}


def _write_realization_inputs(directory, aggregate, hotspots, seed):
    _save_csv(pd.DataFrame({"particle_id": np.arange(aggregate.n_particles),
                            "x_nm": aggregate.positions_nm[:, 0], "y_nm": aggregate.positions_nm[:, 1],
                            "radius_nm": aggregate.radii_nm, "coordination": aggregate.degrees}), directory/"particles.csv")
    _save_csv(pd.DataFrame({"particle_i": aggregate.edges[:, 0], "particle_j": aggregate.edges[:, 1],
                            "surface_gap_nm": aggregate.edge_gaps_nm}), directory/"edges.csv")
    _save_csv(pd.DataFrame({"hotspot_id": np.arange(len(hotspots.pairs)),
                            "particle_i": hotspots.pairs[:, 0], "particle_j": hotspots.pairs[:, 1],
                            "x_nm": hotspots.positions_nm[:, 0], "y_nm": hotspots.positions_nm[:, 1],
                            "gap_nm": hotspots.gaps_nm}), directory/"hotspots.csv")
    config = frozen_configuration() | {"random_seed": int(seed)}
    (directory/"configuration.json").write_text(json.dumps(config, indent=2, default=_jsonable) + "\n")


def run_realization(seed: int, output_dir: Path, calibration=None):
    """Generate, solve, calibrate, and persist one independent realization."""
    calibration = calibration or ComplexGapCalibration()
    aggregate = generate_aggregate(replace(BASE_GEOMETRY, seed=int(seed)))
    hotspots = construct_hotspots(aggregate, 1.0, 6.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_realization_inputs(output_dir, aggregate, hotspots, seed)

    n = aggregate.n_particles
    positions_m = np.c_[aggregate.positions_nm, np.zeros(n)] * 1e-9
    source_particle = int(np.argmin(aggregate.positions_nm[:, 0]))
    center_nm = aggregate.positions_nm[source_particle]
    center_m = np.r_[center_nm, 0.0] * 1e-9
    e0 = 1.0 + 0j
    k = 2*np.pi / 633e-9
    incident_particles = gaussian_incident_field(positions_m, center_m, 55e-9, e0,
                                                  np.array([1., 0., 0.]))
    alpha = sphere_polarizability(aggregate.radii_nm*1e-9, 633e-9, -15+1j)
    solution = solve_coupled_dipoles(positions_m, alpha, incident_particles, k)
    inc, scat, total = fields_at_hotspots(hotspots, positions_m, solution.dipoles_c_m, k,
                                           center_m, 55e-9, e0, np.array([1., 0., 0.]))
    m2_inc = np.sum(abs(inc)**2, axis=1)
    m2_scat = np.sum(abs(scat)**2, axis=1)
    m2_cda = np.sum(abs(total)**2, axis=1)
    m4_cda = m2_cda**2
    euclidean = np.linalg.norm(hotspots.positions_nm-center_nm, axis=1)
    classification = classify_hotspots(m2_inc, euclidean, 1e-6, 500.)
    source_ids = np.flatnonzero(classification == "source")
    launch = int(source_ids[np.argmax(m4_cda[source_ids])])
    steps, geodesic = hotspot_distances(hotspots, np.array([launch]))
    dominance = np.divide(m2_scat, m2_inc, out=np.full_like(m2_inc, np.inf), where=m2_inc > 0)

    delta = aggregate.positions_nm[hotspots.pairs[:, 1]]-aggregate.positions_nm[hotspots.pairs[:, 0]]
    unit = np.c_[delta/np.linalg.norm(delta, axis=1)[:, None], np.zeros(len(delta))]
    scalar_parallel = np.sum(total*unit, axis=1)
    parallel_field = scalar_parallel[:, None]*unit
    perpendicular_field = total-parallel_field
    cparallel, cperp, calibrated = calibration.interpolate(hotspots.gaps_nm)
    corrected = cparallel[:, None]*parallel_field + cperp[:, None]*perpendicular_field
    corrected[~calibrated] = np.nan + 1j*np.nan
    m2_fw = np.sum(abs(corrected)**2, axis=1)
    m4_fw = m2_fw**2
    coordination = np.maximum(aggregate.degrees[hotspots.pairs[:, 0]],
                              aggregate.degrees[hotspots.pairs[:, 1]])

    frame = pd.DataFrame({"seed": seed, "hotspot_id": np.arange(len(hotspots.pairs)),
        "particle_i": hotspots.pairs[:, 0], "particle_j": hotspots.pairs[:, 1],
        "gap_nm": hotspots.gaps_nm, "x_nm": hotspots.positions_nm[:, 0], "y_nm": hotspots.positions_nm[:, 1],
        "euclidean_distance_nm": euclidean, "geodesic_distance_nm": geodesic, "geodesic_steps": steps,
        "classification": classification, "plasmon_dominated_remote": (classification == "remote") & (dominance >= 100),
        "coordination": coordination, "coordination_i": aggregate.degrees[hotspots.pairs[:, 0]],
        "coordination_j": aggregate.degrees[hotspots.pairs[:, 1]], "M2_inc": m2_inc, "M2_scat": m2_scat,
        "M2_CDA": m2_cda, "M4_CDA": m4_cda, "Cparallel_real": cparallel.real,
        "Cparallel_imag": cparallel.imag, "Cperp_real": cperp.real, "Cperp_imag": cperp.imag,
        "Cparallel_abs": abs(cparallel), "Cperp_abs": abs(cperp), "M2_FWcorr": m2_fw, "M4_FWcorr": m4_fw,
        "FW_correction_status": np.where(calibrated, "CALIBRATED", "OUTSIDE_CALIBRATION_DOMAIN"),
        **_field_columns("E_CDA", total), **_field_columns("E_FWcorr", corrected)})
    _save_csv(frame, output_dir/"hotspots_corrected.csv")

    # Aggregate morphology is connected by construction, but calculate it rather than assume it.
    seen = set(); sizes = []
    particle_adj = [set() for _ in range(n)]
    for i, j in aggregate.edges:
        particle_adj[int(i)].add(int(j)); particle_adj[int(j)].add(int(i))
    for start in range(n):
        if start in seen: continue
        component = {start}; queue = [start]; seen.add(start)
        for node in queue:
            for nxt in particle_adj[node]:
                if nxt not in seen: seen.add(nxt); component.add(nxt); queue.append(nxt)
        sizes.append(len(component))
    morphology = {"seed": seed, "number_particles": n, "hotspot_count": len(frame),
        "connected_component_count": len(sizes), "largest_component_fraction": max(sizes)/n,
        "mean_coordination": aggregate.degrees.mean(),
        "coordination_distribution": json.dumps(np.bincount(aggregate.degrees).tolist()),
        "mean_surface_gap_nm": hotspots.gaps_nm.mean(), "minimum_surface_gap_nm": hotspots.gaps_nm.min(),
        "maximum_surface_gap_nm": hotspots.gaps_nm.max(), "aggregate_x_extent_nm": np.ptp(aggregate.positions_nm[:, 0]),
        "aggregate_y_extent_nm": np.ptp(aggregate.positions_nm[:, 1]),
        "source_hotspots": np.sum(classification == "source"), "transition_hotspots": np.sum(classification == "transition"),
        "remote_hotspots": np.sum(classification == "remote"), "CDA_residual": solution.residual_relative,
        "matrix_condition_estimate": np.nan, "matrix_finite": np.isfinite(solution.matrix).all(),
        "dipoles_finite": np.isfinite(solution.dipoles_c_m).all(), "hotspot_fields_finite": np.isfinite(total).all()}
    remote = classification == "remote"
    strongest_remote_cda = int(frame.loc[remote, "M4_CDA"].idxmax())
    coverage = {"seed": seed, "total_hotspot_count": len(frame), "calibrated_hotspot_count": calibrated.sum(),
        "below_domain_hotspot_count": np.sum(hotspots.gaps_nm < calibration.gaps[0]),
        "above_domain_hotspot_count": np.sum(hotspots.gaps_nm > calibration.gaps[-1]),
        "calibrated_fraction": calibrated.mean(), "remote_hotspot_calibrated_fraction": calibrated[remote].mean(),
        "strongest_CDA_remote_hotspot_id": strongest_remote_cda,
        "strongest_CDA_remote_inside_calibration_domain": bool(calibrated[strongest_remote_cda])}
    return frame, morphology, coverage, hotspots.adjacency


def _transition_statistics(seed, frame, adjacency):
    rows = []
    transitions = directed_transitions(adjacency, frame.geodesic_distance_nm.to_numpy())
    for proxy in ("CDA", "FWcorr"):
        values = frame[f"M4_{proxy}"].to_numpy()
        for region in ("whole_network", "remote"):
            pairs = [(u, v) for u, v in transitions if region == "whole_network" or frame.classification.iloc[v] == "remote"]
            if proxy == "FWcorr":
                pairs = [(u, v) for u, v in pairs if np.isfinite(values[u]) and np.isfinite(values[v])]
            ratios = np.array([gamma_de(values[u], values[v]) for u, v in pairs])
            rows.append({"seed": seed, "proxy": proxy, "region": region, "transition_count": len(ratios),
                         "P_Gamma_DE_gt_1": np.mean(ratios > 1) if len(ratios) else np.nan,
                         "P_Gamma_DE_ge_2": np.mean(ratios >= 2) if len(ratios) else np.nan,
                         "P_Gamma_DE_ge_10": np.mean(ratios >= 10) if len(ratios) else np.nan,
                         "P_Gamma_DE_ge_100": np.mean(ratios >= 100) if len(ratios) else np.nan})
    return rows


def _remote_statistics(seed, frame):
    remote = frame.classification == "remote"; source = frame.classification == "source"
    cda_id = frame.loc[remote, "M4_CDA"].idxmax(); valid_remote = remote & frame.M4_FWcorr.notna()
    fw_id = frame.loc[valid_remote, "M4_FWcorr"].idxmax()
    row = frame.loc[fw_id]
    maxima = {"seed": seed, "strongest_remote_hotspot_id_CDA": int(cda_id),
        "strongest_remote_hotspot_id_FWcorr": int(fw_id), "particle_i": int(row.particle_i), "particle_j": int(row.particle_j),
        "gap_nm": row.gap_nm, "x_nm": row.x_nm, "y_nm": row.y_nm, "euclidean_distance_nm": row.euclidean_distance_nm,
        "geodesic_distance_nm": row.geodesic_distance_nm, "coordination": row.coordination,
        "E_CDA_abs": np.sqrt(row.M2_CDA), "Cparallel_abs": row.Cparallel_abs, "Cperp_abs": row.Cperp_abs,
        "M4_CDA_at_FW_max": row.M4_CDA, "M4_FWcorr": row.M4_FWcorr,
        "M4_remote_max_CDA": frame.loc[remote, "M4_CDA"].max(), "M4_remote_max_FWcorr": frame.loc[valid_remote, "M4_FWcorr"].max(),
        "M4_source_max_CDA": frame.loc[source, "M4_CDA"].max(), "M4_source_max_FWcorr": frame.loc[source, "M4_FWcorr"].max()}
    maxima["eta_RS_CDA"] = maxima["M4_remote_max_CDA"]/maxima["M4_source_max_CDA"]
    maxima["eta_RS_FWcorr"] = maxima["M4_remote_max_FWcorr"]/maxima["M4_source_max_FWcorr"]
    valid = frame.loc[valid_remote, ["hotspot_id", "M4_CDA", "M4_FWcorr"]].copy()
    valid["rank_CDA"] = valid.M4_CDA.rank(ascending=False, method="min")
    valid["rank_FWcorr"] = valid.M4_FWcorr.rank(ascending=False, method="min")
    delta = valid.rank_CDA-valid.rank_FWcorr
    ranking = {"seed": seed, "remote_calibrated_count": len(valid),
        "spearman_rank_correlation": spearmanr(valid.M4_CDA, valid.M4_FWcorr).statistic,
        "top5_overlap": len(set(valid.nlargest(5, "M4_CDA").hotspot_id) & set(valid.nlargest(5, "M4_FWcorr").hotspot_id)),
        "top10_overlap": len(set(valid.nlargest(10, "M4_CDA").hotspot_id) & set(valid.nlargest(10, "M4_FWcorr").hotspot_id)),
        "strongest_hotspot_identity_changes": int(cda_id) != int(fw_id),
        "largest_rank_increase": delta.max(), "largest_rank_decrease": delta.min()}
    return maxima, ranking


def _quantile_envelope(all_hotspots):
    data = all_hotspots.copy()
    data["geodesic_bin"] = pd.cut(data.geodesic_distance_nm, GEODESIC_BIN_EDGES_NM, right=False)
    rows = []
    for proxy in ("CDA", "FWcorr"):
        data["log_value"] = np.log10(data[f"M4_{proxy}"].where(data[f"M4_{proxy}"] > 0))
        for interval, group in data.groupby("geodesic_bin", observed=True):
            pooled = group.log_value.dropna()
            seed_quantiles = group.dropna(subset=["log_value"]).groupby("seed").log_value.quantile(QUANTILES).unstack()
            for method, values in (("pooled_hotspot", pooled.quantile(QUANTILES)),
                                   ("realization_balanced", seed_quantiles.mean(axis=0))):
                rows.append({"proxy": proxy, "weighting": method, "bin_min_nm": interval.left,
                    "bin_max_nm": interval.right, "seed_count": group.seed.nunique(), "hotspot_count": len(pooled),
                    **{f"Q{int(q*100):02d}_log10_M4": values.get(q, np.nan) for q in QUANTILES}})
    return pd.DataFrame(rows)


def _correlations(all_hotspots, maxima):
    remote = all_hotspots[(all_hotspots.classification == "remote") & all_hotspots.M4_FWcorr.notna()].copy()
    remote["log10_M4_FWcorr"] = np.log10(remote.M4_FWcorr)
    remote["E_CDA_abs"] = np.sqrt(remote.M2_CDA)
    variables = ["gap_nm", "geodesic_distance_nm", "euclidean_distance_nm", "coordination",
                 "E_CDA_abs", "Cparallel_abs", "Cperp_abs"]
    rows = []
    for variable in variables:
        pooled = spearmanr(remote.log10_M4_FWcorr, remote[variable], nan_policy="omit")
        seed_rho = remote.groupby("seed").apply(lambda x: spearmanr(x.log10_M4_FWcorr, x[variable], nan_policy="omit").statistic,
                                                 include_groups=False)
        strongest = spearmanr(np.log10(maxima.M4_remote_max_FWcorr), maxima[variable] if variable in maxima else np.nan,
                              nan_policy="omit") if variable in maxima else None
        rows.append({"population": "all_remote_hotspots", "variable": variable, "weighting": "pooled_hotspot",
                     "spearman_rho": pooled.statistic, "p_value": pooled.pvalue, "sample_count": len(remote)})
        rows.append({"population": "all_remote_hotspots", "variable": variable, "weighting": "realization_balanced_mean_rho",
                     "spearman_rho": seed_rho.mean(), "p_value": np.nan, "sample_count": seed_rho.notna().sum()})
        if strongest is not None:
            rows.append({"population": "strongest_remote_per_realization", "variable": variable, "weighting": "one_per_seed",
                         "spearman_rho": strongest.statistic, "p_value": strongest.pvalue, "sample_count": len(maxima)})
    return pd.DataFrame(rows)


def _figures(out, morphology, maxima, downstream, ranking, coverage, envelope):
    def save(fig, name):
        fig.tight_layout(); fig.savefig(out/name, format="svg"); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, col, title in zip(axes.flat, ["mean_surface_gap_nm", "mean_coordination", "hotspot_count", "remote_hotspots"],
                              ["mean gap (nm)", "mean coordination", "hotspot count", "remote-hotspot count"]):
        ax.hist(morphology[col], bins=min(7, len(morphology))); ax.set_xlabel(title); ax.set_ylabel("realizations")
    save(fig, "morphology_distributions.svg")
    for col, name, label in [("M4_remote_max_FWcorr", "remote_max_M4_distribution.svg", "log10 remote maximum M4 FWcorr"),
                             ("eta_RS_FWcorr", "remote_source_ratio_distribution.svg", "log10 eta_RS FWcorr")]:
        fig, ax = plt.subplots(); ax.hist(np.log10(maxima[col]), bins=7); ax.set(xlabel=label, ylabel="realizations"); save(fig, name)
    env = envelope[(envelope.proxy == "FWcorr") & (envelope.weighting == "realization_balanced")]
    x = (env.bin_min_nm+env.bin_max_nm)/2
    fig, ax = plt.subplots(); ax.fill_between(x, env.Q05_log10_M4, env.Q95_log10_M4, alpha=.2, label="Q05-Q95")
    ax.fill_between(x, env.Q25_log10_M4, env.Q75_log10_M4, alpha=.35, label="Q25-Q75"); ax.plot(x, env.Q50_log10_M4, label="Q50")
    ax.set(xlabel="geodesic distance (nm)", ylabel="log10 M4 FWcorr", title="Realization-balanced envelope"); ax.legend(); save(fig, "geodesic_M4_ensemble_envelope.svg")
    subset = downstream[(downstream.proxy == "FWcorr")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, threshold in zip(axes, (2, 10, 100)):
        col = f"P_Gamma_DE_ge_{threshold}"
        for region, marker in (("whole_network", "o"), ("remote", "x")):
            s = subset[subset.region == region]; ax.scatter(s.seed.astype(str), s[col], marker=marker, label=region)
        ax.set(title=f"Gamma_DE >= {threshold}", xlabel="seed"); ax.tick_params(axis="x", rotation=90)
    axes[0].set_ylabel("probability"); axes[-1].legend(); save(fig, "downstream_probability_distribution.svg")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4)); axes[0].bar(ranking.seed.astype(str), ranking.spearman_rank_correlation)
    axes[0].tick_params(axis="x", rotation=90); axes[0].set(title="Remote rank stability", ylabel="Spearman rho")
    changed = ranking.strongest_hotspot_identity_changes.value_counts().reindex([False, True], fill_value=0)
    axes[1].bar(["unchanged", "changed"], changed); save(fig, "ranking_stability.svg")
    fig, ax = plt.subplots(); ax.bar(coverage.seed.astype(str), coverage.calibrated_fraction, label="all")
    ax.scatter(coverage.seed.astype(str), coverage.remote_hotspot_calibrated_fraction, c="red", label="remote"); ax.tick_params(axis="x", rotation=90)
    ax.set(ylabel="calibrated fraction", ylim=(0, 1.05)); ax.legend(); save(fig, "calibration_coverage.svg")


def convergence_metrics(maxima, downstream, envelope, sizes=SUPPORTED_ENSEMBLE_SIZES):
    """Reusable prefix convergence framework; callers choose available sizes."""
    rows = []
    ordered = maxima.sort_values("seed")
    for size in sizes:
        if size > len(ordered):
            continue
        sample = ordered.iloc[:size]; seeds = set(sample.seed)
        ds = downstream[(downstream.seed.isin(seeds)) & (downstream.proxy == "FWcorr") & (downstream.region == "whole_network")]
        env = envelope[(envelope.proxy == "FWcorr") & (envelope.weighting == "realization_balanced")]
        rows.append({"N": size, "median_log10_M4_remote_max_FWcorr": np.median(np.log10(sample.M4_remote_max_FWcorr)),
            "median_log10_eta_RS_FWcorr": np.median(np.log10(sample.eta_RS_FWcorr)),
            "mean_P_Gamma_DE_ge_10": ds.P_Gamma_DE_ge_10.mean(), "mean_P_Gamma_DE_ge_100": ds.P_Gamma_DE_ge_100.mean(),
            "representative_geodesic_Q50_log10_M4": env.Q50_log10_M4.median(),
            "representative_geodesic_Q95_log10_M4": env.Q95_log10_M4.median()})
    return pd.DataFrame(rows)


def _summary(out, morphology, coverage, maxima, downstream, ranking, correlations):
    ds = downstream[(downstream.proxy == "FWcorr")]
    coverage_bad = (coverage.remote_hotspot_calibrated_fraction < .9).any() or (~coverage.strongest_CDA_remote_inside_calibration_domain).any()
    unstable = (~morphology[["matrix_finite", "dipoles_finite", "hotspot_fields_finite"]].all(axis=1)).any() or (morphology.CDA_residual > 1e-8).any()
    recommendation = "FIX_PIPELINE_FIRST" if unstable else ("EXTEND_CALIBRATION_FIRST" if coverage_bad else "PROCEED_TO_50")
    probability_lines = []
    for region, group in ds.groupby("region"):
        for column in ("P_Gamma_DE_gt_1", "P_Gamma_DE_ge_2", "P_Gamma_DE_ge_10", "P_Gamma_DE_ge_100"):
            values = group[column].dropna()
            probability_lines.append(f"   - {region}, {column}: mean {values.mean():.4f}, median {values.median():.4f}, range {values.min():.4f}-{values.max():.4f}.")
    lines = ["# Level 4A 10-realization pilot summary", "",
        "The network is a **2D stochastic retarded coupled-dipole approximation (CDA) model** with frozen full-wave MiePy/GMMT isolated-gap calibration. Corrected M4 is a full-wave-calibrated electromagnetic SERS proxy, not experimental Raman intensity.", "",
        f"1. **Exact seeds:** {', '.join(map(str, PILOT_SEEDS))}.",
        "2. **Frozen configuration:** see `FROZEN_CONFIGURATION.json`; only `random_seed` varies.",
        f"3. **Successful realizations:** {len(morphology)}.", "4. **Failed realizations:** 0; no failure reasons.",
        f"5. **Calibration coverage:** all-hotspot range {coverage.calibrated_fraction.min():.3f}-{coverage.calibrated_fraction.max():.3f}; remote range {coverage.remote_hotspot_calibrated_fraction.min():.3f}-{coverage.remote_hotspot_calibrated_fraction.max():.3f}.",
        f"6. **Remote maximum M4 FWcorr:** median {maxima.M4_remote_max_FWcorr.median():.6g}; range {maxima.M4_remote_max_FWcorr.min():.6g}-{maxima.M4_remote_max_FWcorr.max():.6g}.",
        f"7. **eta_RS FWcorr:** median {maxima.eta_RS_FWcorr.median():.6g}; range {maxima.eta_RS_FWcorr.min():.6g}-{maxima.eta_RS_FWcorr.max():.6g}.",
        "8. **Downstream probabilities (FWcorr):**"] + probability_lines + [
        f"9. **Strongest remote identity changes:** {ranking.strongest_hotspot_identity_changes.sum()}/{len(ranking)} realizations.",
        f"10. **Remote rank correlation:** mean {ranking.spearman_rank_correlation.mean():.4f}, median {ranking.spearman_rank_correlation.median():.4f}, range {ranking.spearman_rank_correlation.min():.4f}-{ranking.spearman_rank_correlation.max():.4f}.",
        "11. **Geodesic envelope:** fixed 100 nm bins show quantile propagation structure; no single exponential attenuation coefficient was fitted.",
        "12. **Correlations:** pooled, realization-balanced, and strongest-per-realization exploratory Spearman results are recorded in `data/remote_hotspot_correlations.csv`; they are associations, not causal evidence.",
        "13. **Pilot size:** 10 realizations are insufficient for a final ensemble claim; convergence code supports later N=10,20,30,40,50,75,100.",
        f"14. **Calibration coverage adequate:** {'no' if coverage_bad else 'yes for this pilot'}; outside-domain CDA values are retained without fabricated corrected values.",
        f"15. **Numerical instability:** {'detected' if unstable else 'none detected from residual and finite-value diagnostics'}.",
        f"16. **Recommendation: `{recommendation}`.**", "", "No 50-realization run and no Level 4B work were performed."]
    (out/"LEVEL4A_PILOT_SUMMARY.md").write_text("\n".join(lines)+"\n")


def run_pilot(output_dir="results/level4A", seeds=PILOT_SEEDS):
    """Execute exactly the requested deterministic pilot ensemble."""
    verify_level3b()
    seeds = tuple(map(int, seeds))
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("Level 4A pilot requires exactly 10 unique deterministic seeds")
    out = Path(output_dir); data_dir = out/"data"; data_dir.mkdir(parents=True, exist_ok=True)
    frozen = frozen_configuration()
    (out/"FROZEN_CONFIGURATION.json").write_text(json.dumps(frozen, indent=2)+"\n")
    (out/"FROZEN_CONFIGURATION.md").write_text(frozen_configuration_markdown(frozen))
    _save_csv(pd.DataFrame({"realization": np.arange(1, 11), "seed": seeds}), data_dir/"pilot_seeds.csv")
    calibration = ComplexGapCalibration()
    frames=[]; morphologies=[]; coverages=[]; downstream=[]; maxima=[]; rankings=[]
    for seed in seeds:
        frame, morphology, coverage, adjacency = run_realization(seed, out/"realizations"/f"seed_{seed}", calibration)
        frames.append(frame); morphologies.append(morphology); coverages.append(coverage)
        downstream.extend(_transition_statistics(seed, frame, adjacency))
        maximum, ranking = _remote_statistics(seed, frame); maxima.append(maximum); rankings.append(ranking)
    all_hotspots = pd.concat(frames, ignore_index=True)
    morphology = pd.DataFrame(morphologies); coverage = pd.DataFrame(coverages)
    downstream = pd.DataFrame(downstream); maxima = pd.DataFrame(maxima); ranking = pd.DataFrame(rankings)
    envelope = _quantile_envelope(all_hotspots); correlations = _correlations(all_hotspots, maxima)
    for frame, name in [(morphology, "morphology_summary.csv"), (coverage, "calibration_coverage.csv"),
        (downstream, "downstream_statistics_per_realization.csv"), (maxima, "remote_maxima_per_realization.csv"),
        (ranking, "ranking_statistics_per_realization.csv"), (all_hotspots, "all_hotspots_ensemble.csv"),
        (envelope, "geodesic_quantile_envelope.csv"), (correlations, "remote_hotspot_correlations.csv")]:
        _save_csv(frame, data_dir/name)
    _figures(out, morphology, maxima, downstream, ranking, coverage, envelope)
    _summary(out, morphology, coverage, maxima, downstream, ranking, correlations)
    return {"successful_realizations": len(frames), "failed_realizations": 0,
            "output_dir": str(out), "seeds": seeds}


if __name__ == "__main__":
    result = run_pilot()
    print("LEVEL 4A PILOT COMPLETE" if result["successful_realizations"] == 10 else "LEVEL 4A PILOT INCOMPLETE")
