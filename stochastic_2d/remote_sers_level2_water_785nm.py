# %% [markdown]
# Final Level 2 — 785 nm excitation, Ag in water, Levels 2.6–2.8
#
# This single-file script is based on the current Codex/GitHub implementation
# in zyt2025728/remote-SERS-model (Level 2.6).  It embeds the validated geometry
# generator, Gaussian local excitation, retarded 3D dyadic Green tensor,
# radiatively corrected spherical polarizability, full all-pair CDA solve,
# per-edge gap-center fields, and a two-frequency electromagnetic SERS proxy.
# The geometry/statistics are unchanged.  The optical layer uses wavelength-
# resolved Ag permittivity from the accompanying Johnson data table and the
# wavelength-dependent refractive index of liquid water at 19 degrees C.
#
# It can be run as a normal .py file or cell-by-cell in Jupyter/VS Code because
# it uses Jupytext-style '# %%' cell markers.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import math
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm


# =============================================================================
# USER SETTINGS — CHANGE THESE FIRST
# =============================================================================
# "baseline": reproduce the current N=200 Level-2.6 realization only
# "pilot"   : run N=200, 400, 800 with one seed each
# "full"    : run N=200, 400, 800 with 10 seeds each (can be very slow locally)
RUN_MODE = "full"

BASE_SEED = 20260813
NETWORK_SIZES = [200, 400, 800]
FULL_NUMBER_OF_SEEDS = 10

# Optical configuration for this requested recalculation
WAVELENGTH_NM = 785.0
# Configurable representative Raman band. Replace with the measured Raman shift
# when a specific analyte band is available.
RAMAN_SHIFT_CM1 = 1000.0
# Supplied wavelength-resolved Johnson Ag dielectric table.  The code linearly
# interpolates eps_real and eps_imag at both excitation and Stokes wavelengths.
AG_DIELECTRIC_CSV = Path(__file__).resolve().parent / "Ag_Johnson_eps.csv"
INCIDENT_AMPLITUDE_V_PER_M = 1.0 + 0.0j
BEAM_WAIST_NM = 55.0
POLARIZATION = np.array([1.0, 0.0, 0.0])
REMOTE_INCIDENT_THRESHOLD = 1e-6
REMOTE_DISTANCE_NM = 500.0
MINIMUM_GRAPH_STEPS = 15

# Optional Level-3B magnitude calibration table. When supplied, it must contain:
# gap_nm,C_exc_abs,C_raman_abs
# The factors multiply the complex CDA gap field before |E|^2 is evaluated.
# None means C_local=1 and every result is labelled "uncalibrated CDA".
LOCAL_FIELD_CALIBRATION_CSV = None

# Propagation-metric processing
SMOOTHING_WINDOW = 5  # odd integer; smoothing is in log10(M/M_source)

# Output location: Desktop if it exists, otherwise the current working directory
_desktop = Path.home() / "Desktop"
OUTPUT_ROOT = (_desktop if _desktop.exists() else Path.cwd()) / "remote_SERS_Level2_Water_785nm_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Figure handling
SHOW_SUMMARY_FIGURES = False
SAVE_PDF = True


# =============================================================================
# 1. GEOMETRY — exact current Level-2.6 growth logic
# =============================================================================
@dataclass(frozen=True)
class GeometryConfig:
    n_particles: int = 200
    radius_mean_nm: float = 10.0
    radius_std_nm: float = 0.0
    radius_min_nm: float = 10.0
    radius_max_nm: float = 10.0
    gap_mean_nm: float = 3.0
    gap_std_nm: float = 1.0
    gap_min_nm: float = 1.0
    gap_max_nm: float = 6.0
    positional_disorder_rad: float = 0.45
    connection_gap_nm: float = 6.0
    seed: int = 20260813
    max_attempts_per_particle: int = 4000


@dataclass(frozen=True)
class Aggregate:
    positions_nm: np.ndarray
    radii_nm: np.ndarray
    edges: np.ndarray
    edge_gaps_nm: np.ndarray
    seed: int

    @property
    def n_particles(self) -> int:
        return len(self.radii_nm)

    @property
    def degrees(self) -> np.ndarray:
        values = np.zeros(self.n_particles, dtype=np.int64)
        if len(self.edges):
            np.add.at(values, self.edges[:, 0], 1)
            np.add.at(values, self.edges[:, 1], 1)
        return values


def _truncated_normal(rng, mean, std, low, high):
    if std < 0 or low > high:
        raise ValueError("invalid distribution parameters")
    if std == 0:
        if not low <= mean <= high:
            raise ValueError("constant distribution lies outside its bounds")
        return float(mean)
    for _ in range(10_000):
        value = float(rng.normal(mean, std))
        if low <= value <= high:
            return value
    raise RuntimeError("could not sample the requested truncated distribution")


def generate_aggregate(config: GeometryConfig) -> Aggregate:
    """Grow the same connected, non-overlapping stochastic 2D aggregate as Level 2.6."""
    if config.n_particles < 1:
        raise ValueError("n_particles must be positive")
    if config.connection_gap_nm < config.gap_max_nm:
        raise ValueError("connection_gap_nm must cover every generated attachment gap")

    rng = np.random.default_rng(config.seed)
    radii = np.array([
        _truncated_normal(
            rng,
            config.radius_mean_nm,
            config.radius_std_nm,
            config.radius_min_nm,
            config.radius_max_nm,
        )
        for _ in range(config.n_particles)
    ])

    positions = np.zeros((config.n_particles, 2), dtype=float)
    attachment_edges = []
    degrees = np.zeros(config.n_particles, dtype=int)

    for child in range(1, config.n_particles):
        placed = False
        for attempt in range(config.max_attempts_per_particle):
            existing = np.arange(child)

            # Current Codex logic:
            # most growth favors low-degree nodes (branching);
            # every fifth particle favors locations near the current centroid.
            if child % 5:
                weights = 1.0 / (1.0 + degrees[:child])
            else:
                centroid = positions[:child].mean(axis=0)
                distance = np.linalg.norm(positions[:child] - centroid, axis=1)
                weights = 1.0 / (1.0 + distance)

            parent = int(rng.choice(existing, p=weights / weights.sum()))
            base_angle = math.atan2(positions[parent, 1], positions[parent, 0])

            if attempt < 12 and child > 2:
                angle = base_angle + rng.normal(0.0, config.positional_disorder_rad)
            else:
                angle = rng.uniform(-math.pi, math.pi)

            gap = _truncated_normal(
                rng,
                config.gap_mean_nm,
                config.gap_std_nm,
                config.gap_min_nm,
                config.gap_max_nm,
            )
            separation = radii[parent] + radii[child] + gap
            candidate = positions[parent] + separation * np.array([math.cos(angle), math.sin(angle)])

            center_distances = np.linalg.norm(positions[:child] - candidate, axis=1)
            required = radii[:child] + radii[child] + config.gap_min_nm
            if np.all(center_distances >= required - 1e-10):
                positions[child] = candidate
                attachment_edges.append((parent, child))
                degrees[parent] += 1
                degrees[child] += 1
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"failed to place particle {child}; try another seed or relax geometry settings"
            )

    # Infer ALL physical near contacts after growth; these edges define the geometry/path graph.
    edges = []
    gaps = []
    for i in range(config.n_particles):
        offsets = positions[i + 1:] - positions[i]
        surface_gaps = np.linalg.norm(offsets, axis=1) - radii[i] - radii[i + 1:]
        for relative_j in np.flatnonzero(surface_gaps <= config.connection_gap_nm + 1e-9):
            j = i + 1 + int(relative_j)
            edges.append((i, j))
            gaps.append(max(0.0, float(surface_gaps[relative_j])))

    aggregate = Aggregate(
        positions_nm=positions,
        radii_nm=radii,
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        edge_gaps_nm=np.asarray(gaps, dtype=float),
        seed=config.seed,
    )

    attachment_set = {tuple(sorted(edge)) for edge in attachment_edges}
    inferred_set = {tuple(edge) for edge in aggregate.edges}
    if not attachment_set.issubset(inferred_set):
        raise RuntimeError("internal error: connected backbone was lost")
    return aggregate


def minimum_surface_gap(aggregate: Aggregate) -> float:
    minimum = math.inf
    for i in range(aggregate.n_particles - 1):
        distances = np.linalg.norm(
            aggregate.positions_nm[i + 1:] - aggregate.positions_nm[i], axis=1
        )
        gaps = distances - aggregate.radii_nm[i] - aggregate.radii_nm[i + 1:]
        minimum = min(minimum, float(gaps.min()))
    return minimum


# =============================================================================
# 2. LOCAL GAUSSIAN EXCITATION
# =============================================================================
def gaussian_incident_field(
    positions_m,
    center_m,
    waist_m,
    amplitude_v_per_m=1.0,
    polarization=None,
):
    if waist_m <= 0:
        raise ValueError("waist_m must be positive")
    pol = np.array([1.0, 0.0, 0.0]) if polarization is None else np.asarray(polarization, dtype=float)
    if pol.shape != (3,) or np.linalg.norm(pol) == 0:
        raise ValueError("polarization must be a nonzero 3-vector")
    pol = pol / np.linalg.norm(pol)
    radial_squared = np.sum((np.asarray(positions_m) - np.asarray(center_m)) ** 2, axis=1)
    envelope = amplitude_v_per_m * np.exp(-radial_squared / waist_m**2)
    return envelope[:, None] * pol[None, :]


# =============================================================================
# 3. FULL RETARDED 3D DYADIC GREEN TENSOR
# =============================================================================
EPSILON_0 = 8.8541878128e-12


def electric_dyadic(displacement_m, wave_number, medium_relative_permittivity=1.0):
    vector = np.asarray(displacement_m, dtype=float)
    distance = float(np.linalg.norm(vector))
    if vector.shape != (3,) or distance == 0 or not np.isfinite(distance):
        raise ValueError("displacement must be a finite, nonzero 3-vector")

    direction = vector / distance
    nn = np.outer(direction, direction)
    identity = np.eye(3)
    kr = wave_number * distance

    tensor = (
        (wave_number**2 / distance) * (identity - nn)
        + (1 / distance**3 - 1j * wave_number / distance**2) * (3 * nn - identity)
    )
    return (
        np.exp(1j * kr)
        * tensor
        / (4 * np.pi * EPSILON_0 * medium_relative_permittivity)
    )


# =============================================================================
# 4. PARTICLE POLARIZABILITY
# =============================================================================
def sphere_polarizability(
    radii_m,
    wavelength_m,
    particle_relative_permittivity,
    medium_relative_permittivity=1.0,
):
    radii = np.asarray(radii_m, dtype=float)
    if np.any(radii <= 0) or wavelength_m <= 0:
        raise ValueError("radii and wavelength must be positive")

    contrast = (
        (particle_relative_permittivity - medium_relative_permittivity)
        / (particle_relative_permittivity + 2 * medium_relative_permittivity)
    )
    alpha_static = (
        4
        * np.pi
        * EPSILON_0
        * medium_relative_permittivity
        * radii**3
        * contrast
    )
    k = 2 * np.pi * np.sqrt(medium_relative_permittivity) / wavelength_m
    return alpha_static / (
        1
        - 1j
        * k**3
        * alpha_static
        / (6 * np.pi * EPSILON_0 * medium_relative_permittivity)
    )


# =============================================================================
# 5. FULL ALL-PAIR RETARDED CDA SOLVER
# =============================================================================
def assemble_matrix(positions_m, polarizabilities, wave_number, medium_relative_permittivity=1.0):
    positions = np.asarray(positions_m, dtype=float)
    n = len(positions)
    alphas = np.asarray(polarizabilities)
    if positions.shape != (n, 3) or alphas.shape != (n,):
        raise ValueError("positions must be (N,3) and polarizabilities must be (N,)")

    matrix = np.eye(3 * n, dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                green = electric_dyadic(
                    positions[i] - positions[j],
                    wave_number,
                    medium_relative_permittivity,
                )
                matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -alphas[i] * green
    return matrix


def solve_coupled_dipoles(
    positions_m,
    polarizabilities,
    incident_field,
    wave_number,
    medium_relative_permittivity=1.0,
):
    matrix = assemble_matrix(
        positions_m,
        polarizabilities,
        wave_number,
        medium_relative_permittivity,
    )
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("coupled-dipole matrix contains NaN or infinity")

    rhs = (
        np.asarray(polarizabilities)[:, None] * np.asarray(incident_field)
    ).reshape(-1)
    solution = np.linalg.solve(matrix, rhs)

    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("coupled-dipole solution contains NaN or infinity")

    residual = np.linalg.norm(matrix @ solution - rhs) / max(
        np.linalg.norm(rhs), np.finfo(float).tiny
    )
    return solution.reshape(-1, 3), float(residual)


# =============================================================================
# 5B. GAP-CENTER FIELDS AND TWO-FREQUENCY SERS READOUT
# =============================================================================
def stokes_wavelength_nm(excitation_wavelength_nm, raman_shift_cm1):
    """Convert an excitation wavelength and positive Raman shift to Stokes nm."""
    excitation_wavenumber_cm1 = 1e7 / float(excitation_wavelength_nm)
    stokes_wavenumber_cm1 = excitation_wavenumber_cm1 - float(raman_shift_cm1)
    if stokes_wavenumber_cm1 <= 0:
        raise ValueError("Raman shift must be smaller than the excitation wavenumber")
    return 1e7 / stokes_wavenumber_cm1


def silver_permittivity_from_table(wavelength_nm, csv_path=AG_DIELECTRIC_CSV):
    """Linearly interpolate the supplied Johnson Ag epsilon table.

    Interpolation is performed directly on eps_real and eps_imag.  Extrapolation
    is forbidden so an accidental wavelength outside the measured table fails
    loudly instead of silently changing the material model.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Ag dielectric table not found: {path}. Keep Ag_Johnson_eps.csv "
            "in the same directory as this Python file."
        )

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"lam_nm", "eps_real", "eps_imag"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Ag table must contain lam_nm, eps_real and eps_imag columns"
            )
        for row in reader:
            rows.append(
                (
                    float(row["lam_nm"]),
                    float(row["eps_real"]),
                    float(row["eps_imag"]),
                )
            )

    if len(rows) < 2:
        raise ValueError("Ag dielectric table must contain at least two rows")
    table = np.asarray(sorted(rows), dtype=float)
    wavelengths = table[:, 0]
    wavelength_nm = float(wavelength_nm)
    if wavelength_nm < wavelengths[0] or wavelength_nm > wavelengths[-1]:
        raise ValueError(
            f"{wavelength_nm:.6g} nm is outside the Ag table range "
            f"[{wavelengths[0]:.6g}, {wavelengths[-1]:.6g}] nm"
        )
    eps_real = np.interp(wavelength_nm, wavelengths, table[:, 1])
    eps_imag = np.interp(wavelength_nm, wavelengths, table[:, 2])
    return complex(eps_real, eps_imag)


def water_refractive_index_daimon_19c(wavelength_nm):
    """Real refractive index of water from Daimon & Masumura (2007), 19 C.

    The four-term dispersion relation is valid from 182 to 1129 nm.  This
    Level-2 calculation treats water as lossless at 785 and 851.872 nm.
    """
    wavelength_um = float(wavelength_nm) * 1e-3
    if not 0.182 <= wavelength_um <= 1.129:
        raise ValueError(
            "Daimon water dispersion is only valid from 182 to 1129 nm"
        )
    wavelength_um_sq = wavelength_um**2
    n_squared = 1.0 + (
        5.672526103e-1 * wavelength_um_sq
        / (wavelength_um_sq - 5.085550461e-3)
        + 1.736581125e-1 * wavelength_um_sq
        / (wavelength_um_sq - 1.814938654e-2)
        + 2.121531502e-2 * wavelength_um_sq
        / (wavelength_um_sq - 2.617260739e-2)
        + 1.138493213e-1 * wavelength_um_sq
        / (wavelength_um_sq - 1.073888649e1)
    )
    return float(np.sqrt(n_squared))


def optical_parameters():
    """Return all excitation/Stokes material parameters used by the solver."""
    wavelength_raman_nm = stokes_wavelength_nm(WAVELENGTH_NM, RAMAN_SHIFT_CM1)
    n_water_exc = water_refractive_index_daimon_19c(WAVELENGTH_NM)
    n_water_raman = water_refractive_index_daimon_19c(wavelength_raman_nm)
    return {
        "excitation_wavelength_nm": float(WAVELENGTH_NM),
        "raman_shift_cm1": float(RAMAN_SHIFT_CM1),
        "raman_wavelength_nm": float(wavelength_raman_nm),
        "silver_epsilon_exc": silver_permittivity_from_table(WAVELENGTH_NM),
        "silver_epsilon_raman": silver_permittivity_from_table(
            wavelength_raman_nm
        ),
        "water_n_exc": n_water_exc,
        "water_n_raman": n_water_raman,
        "water_epsilon_exc": complex(n_water_exc**2, 0.0),
        "water_epsilon_raman": complex(n_water_raman**2, 0.0),
    }


def physical_gap_centers_nm(aggregate):
    """Return the physical midpoint between the two facing particle surfaces."""
    centers = np.empty((len(aggregate.edges), 2), dtype=float)
    for e, (i, j) in enumerate(aggregate.edges):
        i, j = int(i), int(j)
        delta = aggregate.positions_nm[j] - aggregate.positions_nm[i]
        unit = delta / np.linalg.norm(delta)
        surface_i = aggregate.positions_nm[i] + aggregate.radii_nm[i] * unit
        surface_j = aggregate.positions_nm[j] - aggregate.radii_nm[j] * unit
        centers[e] = 0.5 * (surface_i + surface_j)
    return centers


def evaluate_total_field(
    evaluation_positions_m,
    particle_positions_m,
    dipoles,
    incident_field_at_evaluation,
    wave_number,
    medium_relative_permittivity=1.0,
):
    """Evaluate E_ext + sum_i G(r-r_i)p_i at arbitrary non-particle points."""
    evaluation_positions_m = np.asarray(evaluation_positions_m, dtype=float)
    particle_positions_m = np.asarray(particle_positions_m, dtype=float)
    dipoles = np.asarray(dipoles, dtype=complex)
    total = np.asarray(incident_field_at_evaluation, dtype=complex).copy()
    for q, point in enumerate(evaluation_positions_m):
        scattered = np.zeros(3, dtype=complex)
        for position, dipole in zip(particle_positions_m, dipoles):
            scattered += electric_dyadic(
                point - position,
                wave_number,
                medium_relative_permittivity,
            ) @ dipole
        total[q] += scattered
    return total


def calibration_factors(edge_gaps_nm):
    """Return Level-3B |C_local| factors interpolated by surface gap.

    The interface intentionally uses magnitude factors because this Level-2.8
    output is an intensity/SERS readout.  Phase-resolved calibration can later
    replace it without changing the transport solve.
    """
    gaps = np.asarray(edge_gaps_nm, dtype=float)
    if LOCAL_FIELD_CALIBRATION_CSV is None:
        return np.ones_like(gaps), np.ones_like(gaps), "uncalibrated_CDA_C_local_equals_1"

    path = Path(LOCAL_FIELD_CALIBRATION_CSV)
    table = np.genfromtxt(path, delimiter=",", names=True)
    required = {"gap_nm", "C_exc_abs", "C_raman_abs"}
    if table.dtype.names is None or not required.issubset(table.dtype.names):
        raise ValueError(
            "Calibration CSV requires columns gap_nm,C_exc_abs,C_raman_abs"
        )
    order = np.argsort(table["gap_nm"])
    x = np.asarray(table["gap_nm"])[order]
    c_exc = np.interp(gaps, x, np.asarray(table["C_exc_abs"])[order])
    c_ram = np.interp(gaps, x, np.asarray(table["C_raman_abs"])[order])
    if np.any(c_exc <= 0) or np.any(c_ram <= 0):
        raise ValueError("All local-field calibration magnitudes must be positive")
    return c_exc, c_ram, f"calibrated_from_{path.name}"


def node_tree_geodesic_nm(aggregate, source, graph_distance, parent):
    """Distance along the same BFS parent tree used by the validated path code."""
    result = np.full(aggregate.n_particles, np.nan, dtype=float)
    result[source] = 0.0
    reachable = np.flatnonzero(graph_distance >= 0)
    for node in reachable[np.argsort(graph_distance[reachable])]:
        if node == source:
            continue
        p = int(parent[node])
        result[node] = result[p] + np.linalg.norm(
            aggregate.positions_nm[node] - aggregate.positions_nm[p]
        )
    return result


# =============================================================================
# 6. GRAPH / GEODESIC PATH ANALYSIS
# =============================================================================
def build_graph(edges, n, source):
    adjacency = [[] for _ in range(n)]
    for i, j in edges:
        i, j = int(i), int(j)
        adjacency[i].append(j)
        adjacency[j].append(i)

    graph_distance = np.full(n, -1, dtype=int)
    parent = np.full(n, -1, dtype=int)
    graph_distance[source] = 0
    queue = [source]

    for u in queue:
        for v in sorted(adjacency[u]):
            if graph_distance[v] < 0:
                graph_distance[v] = graph_distance[u] + 1
                parent[v] = u
                queue.append(v)

    return adjacency, graph_distance, parent


def reconstruct_path(parent, target):
    path = [int(target)]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    return np.asarray(path[::-1], dtype=int)


def smooth_monotone_log_envelope(distance_nm, m_relative, window=5):
    """Smooth log10 intensity then enforce a non-increasing envelope.

    This avoids defining L_1/e or L_1% from one isolated noisy crossing while
    keeping the raw non-monotonic M(d_G) values unchanged in the saved outputs.
    """
    distance_nm = np.asarray(distance_nm, dtype=float)
    m_relative = np.asarray(m_relative, dtype=float)
    tiny = np.finfo(float).tiny
    logm = np.log10(np.maximum(m_relative, tiny))

    if window < 1:
        window = 1
    if window % 2 == 0:
        window += 1
    window = min(window, len(logm) if len(logm) % 2 == 1 else max(1, len(logm) - 1))

    if window > 1:
        half = window // 2
        padded = np.pad(logm, (half, half), mode="edge")
        kernel = np.ones(window) / window
        smooth = np.convolve(padded, kernel, mode="valid")
    else:
        smooth = logm.copy()

    smooth -= smooth[0]  # exactly normalize smoothed envelope to source = 1
    monotone_log = np.minimum.accumulate(smooth)
    return 10 ** monotone_log


def threshold_distance(distance_nm, envelope, threshold):
    distance_nm = np.asarray(distance_nm, dtype=float)
    envelope = np.asarray(envelope, dtype=float)

    if envelope[0] <= threshold:
        return 0.0

    crossed = np.flatnonzero(envelope <= threshold)
    if len(crossed) == 0:
        return float("nan")

    k = int(crossed[0])
    if k == 0:
        return float(distance_nm[0])

    x0, x1 = distance_nm[k - 1], distance_nm[k]
    y0, y1 = np.log10(envelope[k - 1]), np.log10(envelope[k])
    yt = math.log10(threshold)
    if y1 == y0:
        return float(x1)
    fraction = (yt - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def edge_loop_flags(edges, n):
    """Return True for every edge that lies in at least one graph loop.

    An undirected edge is in a loop exactly when it is not a bridge.  The
    Tarjan bridge test below gives the same classification as the original
    Level-2.6 remove-one-edge reachability test, but is much faster for N=800.
    """
    edge_adjacency = [[] for _ in range(n)]
    for edge_id, (i, j) in enumerate(edges):
        i, j = int(i), int(j)
        edge_adjacency[i].append((j, edge_id))
        edge_adjacency[j].append((i, edge_id))

    discovery = np.full(n, -1, dtype=int)
    low = np.full(n, -1, dtype=int)
    bridges = np.zeros(len(edges), dtype=bool)
    clock = 0

    def visit(u, parent_edge):
        nonlocal clock
        discovery[u] = low[u] = clock
        clock += 1
        for v, edge_id in edge_adjacency[u]:
            if edge_id == parent_edge:
                continue
            if discovery[v] < 0:
                visit(v, edge_id)
                low[u] = min(low[u], low[v])
                if low[v] > discovery[u]:
                    bridges[edge_id] = True
            else:
                low[u] = min(low[u], discovery[v])

    for root in range(n):
        if discovery[root] < 0:
            visit(root, -1)
    return ~bridges


def particle_downstream_analysis(aggregate, graph_distance, dipole_intensity,
                                 euclidean_nm):
    """Exact whole-network Level-2.6 downstream particle-edge scan."""
    n = aggregate.n_particles
    incident_gaps = [[] for _ in range(n)]
    for edge_id, (i, j) in enumerate(aggregate.edges):
        gap = float(aggregate.edge_gaps_nm[edge_id])
        incident_gaps[int(i)].append(gap)
        incident_gaps[int(j)].append(gap)
    mean_local_gap = np.array([
        np.mean(values) if values else float("nan") for values in incident_gaps
    ])
    loop_flags = edge_loop_flags(aggregate.edges, n)
    tiny = np.finfo(float).tiny
    rows = []
    for edge_id, (i0, j0) in enumerate(aggregate.edges):
        i0, j0 = int(i0), int(j0)
        if graph_distance[i0] == graph_distance[j0]:
            continue
        if graph_distance[i0] < graph_distance[j0]:
            upstream, downstream = i0, j0
        else:
            upstream, downstream = j0, i0
        ratio = float(dipole_intensity[downstream] /
                      max(float(dipole_intensity[upstream]), tiny))
        rows.append({
            "edge_id": int(edge_id),
            "upstream_id": upstream,
            "downstream_id": downstream,
            "upstream_graph_distance": int(graph_distance[upstream]),
            "downstream_graph_distance": int(graph_distance[downstream]),
            "upstream_coordination": int(aggregate.degrees[upstream]),
            "downstream_coordination": int(aggregate.degrees[downstream]),
            "upstream_mean_local_gap_nm": float(mean_local_gap[upstream]),
            "downstream_mean_local_gap_nm": float(mean_local_gap[downstream]),
            "connecting_gap_nm": float(aggregate.edge_gaps_nm[edge_id]),
            "edge_in_loop": int(loop_flags[edge_id]),
            "downstream_euclidean_nm": float(euclidean_nm[downstream]),
            "enhancement_ratio": ratio,
            "is_enhancement": int(ratio > 1.0),
        })
    return rows


def gap_sers_downstream_analysis(aggregate, gap_dg_nm, gap_euclidean_nm,
                                 gap_sers_ef):
    """Whole-network downstream scan for adjacent physical gap hotspots.

    Gap hotspots are nodes of the line graph: two gaps are adjacent when their
    particle edges share one particle.  They are oriented by increasing BFS-tree
    geodesic coordinate, then the downstream/upstream two-frequency SERS-EF
    ratio is tested against one.  Each unordered gap pair is counted once.
    """
    incident_edges = [[] for _ in range(aggregate.n_particles)]
    for edge_id, (i, j) in enumerate(aggregate.edges):
        incident_edges[int(i)].append(edge_id)
        incident_edges[int(j)].append(edge_id)

    tiny = np.finfo(float).tiny
    seen_pairs = set()
    rows = []
    for shared_particle, edge_ids in enumerate(incident_edges):
        for a in range(len(edge_ids)):
            for b in range(a + 1, len(edge_ids)):
                e0, e1 = int(edge_ids[a]), int(edge_ids[b])
                pair = tuple(sorted((e0, e1)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if np.isclose(gap_dg_nm[e0], gap_dg_nm[e1], rtol=0.0, atol=1e-12):
                    continue
                if gap_dg_nm[e0] < gap_dg_nm[e1]:
                    upstream_gap, downstream_gap = e0, e1
                else:
                    upstream_gap, downstream_gap = e1, e0
                ratio = float(gap_sers_ef[downstream_gap] /
                              max(float(gap_sers_ef[upstream_gap]), tiny))
                rows.append({
                    "shared_particle": int(shared_particle),
                    "shared_particle_coordination": int(
                        aggregate.degrees[shared_particle]
                    ),
                    "upstream_gap_id": upstream_gap,
                    "downstream_gap_id": downstream_gap,
                    "upstream_gap_geodesic_nm": float(gap_dg_nm[upstream_gap]),
                    "downstream_gap_geodesic_nm": float(gap_dg_nm[downstream_gap]),
                    "downstream_gap_euclidean_nm": float(
                        gap_euclidean_nm[downstream_gap]
                    ),
                    "upstream_surface_gap_nm": float(
                        aggregate.edge_gaps_nm[upstream_gap]
                    ),
                    "downstream_surface_gap_nm": float(
                        aggregate.edge_gaps_nm[downstream_gap]
                    ),
                    "upstream_SERS_EF": float(gap_sers_ef[upstream_gap]),
                    "downstream_SERS_EF": float(gap_sers_ef[downstream_gap]),
                    "enhancement_ratio": ratio,
                    "is_enhancement": int(ratio > 1.0),
                })
    return rows


def grouped_probability(rows, key):
    groups = sorted({row[key] for row in rows})
    result = []
    for group in groups:
        selected = [row for row in rows if row[key] == group]
        events = sum(row["is_enhancement"] for row in selected)
        result.append((group, len(selected), events, events / len(selected)))
    return result


# =============================================================================
# 7. CSV / FIGURE HELPERS
# =============================================================================
def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(fig, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    if SAVE_PDF:
        fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


# =============================================================================
# 8. ONE STOCHASTIC NETWORK REALIZATION
# =============================================================================
def run_one_realization(n, seed, out_dir):
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running N={n}, seed={seed} ---")
    t0 = time.perf_counter()

    cfg = GeometryConfig(
        n_particles=n,
        seed=seed,
        radius_mean_nm=10.0,
        radius_std_nm=0.0,
        radius_min_nm=10.0,
        radius_max_nm=10.0,
        gap_mean_nm=3.0,
        gap_std_nm=1.0,
        gap_min_nm=1.0,
        gap_max_nm=6.0,
        positional_disorder_rad=0.45,
        connection_gap_nm=6.0,
    )

    aggregate = generate_aggregate(cfg)
    pos3_m = np.c_[aggregate.positions_nm, np.zeros(n)] * 1e-9

    optics = optical_parameters()
    wavelength_raman_nm = optics["raman_wavelength_nm"]
    silver_epsilon_exc = optics["silver_epsilon_exc"]
    silver_epsilon_raman = optics["silver_epsilon_raman"]
    water_n_exc = optics["water_n_exc"]
    water_n_raman = optics["water_n_raman"]
    water_epsilon_exc = optics["water_epsilon_exc"]
    water_epsilon_raman = optics["water_epsilon_raman"]

    source = int(np.argmin(aggregate.positions_nm[:, 0]))
    source_xy_nm = aggregate.positions_nm[source]

    incident = gaussian_incident_field(
        pos3_m,
        np.r_[source_xy_nm, 0.0] * 1e-9,
        BEAM_WAIST_NM * 1e-9,
        INCIDENT_AMPLITUDE_V_PER_M,
        POLARIZATION,
    )

    # The in-medium wave number is used consistently in the radiative
    # correction, all-pair Green tensor, and arbitrary-point gap evaluation.
    k = 2 * np.pi * water_n_exc / (WAVELENGTH_NM * 1e-9)
    alpha = sphere_polarizability(
        aggregate.radii_nm * 1e-9,
        WAVELENGTH_NM * 1e-9,
        silver_epsilon_exc,
        water_epsilon_exc,
    )

    print("Assembling and solving full all-pair CDA matrix ...")
    dipoles, residual = solve_coupled_dipoles(
        pos3_m,
        alpha,
        incident,
        k,
        water_epsilon_exc,
    )

    # Reciprocity-style two-frequency readout at a representative Stokes band.
    # It uses the same Gaussian illumination/collection geometry at lambda_R.
    incident_raman = gaussian_incident_field(
        pos3_m,
        np.r_[source_xy_nm, 0.0] * 1e-9,
        BEAM_WAIST_NM * 1e-9,
        INCIDENT_AMPLITUDE_V_PER_M,
        POLARIZATION,
    )
    k_raman = 2 * np.pi * water_n_raman / (wavelength_raman_nm * 1e-9)
    alpha_raman = sphere_polarizability(
        aggregate.radii_nm * 1e-9,
        wavelength_raman_nm * 1e-9,
        silver_epsilon_raman,
        water_epsilon_raman,
    )
    print(
        f"Solving reciprocal Raman-wavelength CDA at {wavelength_raman_nm:.2f} nm ..."
    )
    dipoles_raman, residual_raman = solve_coupled_dipoles(
        pos3_m,
        alpha_raman,
        incident_raman,
        k_raman,
        water_epsilon_raman,
    )

    incident_intensity = np.sum(np.abs(incident) ** 2, axis=1)
    dipole_intensity = np.sum(np.abs(dipoles) ** 2, axis=1)
    incident_normalized = incident_intensity / incident_intensity.max()
    dipole_normalized = dipole_intensity / dipole_intensity.max()

    euclidean_nm = np.linalg.norm(aggregate.positions_nm - source_xy_nm, axis=1)
    adjacency, graph_distance, parent = build_graph(aggregate.edges, n, source)

    remote = (
        (incident_normalized < REMOTE_INCIDENT_THRESHOLD)
        & (euclidean_nm > REMOTE_DISTANCE_NM)
    )
    eligible = np.flatnonzero(remote & (graph_distance >= MINIMUM_GRAPH_STEPS))

    if not len(eligible):
        raise RuntimeError(
            f"N={n}, seed={seed}: no remote particle satisfies the current "
            f"Level-2.6 criteria (Euclidean>{REMOTE_DISTANCE_NM} nm, "
            f"incident<{REMOTE_INCIDENT_THRESHOLD}, steps>={MINIMUM_GRAPH_STEPS})."
        )

    remote_indices = np.flatnonzero(remote)
    strongest_remote = int(remote_indices[np.argmax(dipole_intensity[remote])])
    if graph_distance[strongest_remote] >= MINIMUM_GRAPH_STEPS:
        target = strongest_remote
    else:
        target = int(eligible[np.argmax(dipole_intensity[eligible])])

    path = reconstruct_path(parent, target)
    edge_lengths_nm = np.linalg.norm(
        np.diff(aggregate.positions_nm[path], axis=0), axis=1
    )
    cumulative_geodesic_nm = np.r_[0.0, np.cumsum(edge_lengths_nm)]

    # Per-edge physical gap-center fields. These are hotspot observables derived
    # from the already-solved collective particle dipoles; they do not alter CDA.
    gap_xy_nm = physical_gap_centers_nm(aggregate)
    gap_pos3_m = np.c_[gap_xy_nm, np.zeros(len(gap_xy_nm))] * 1e-9
    incident_gap_exc = gaussian_incident_field(
        gap_pos3_m,
        np.r_[source_xy_nm, 0.0] * 1e-9,
        BEAM_WAIST_NM * 1e-9,
        INCIDENT_AMPLITUDE_V_PER_M,
        POLARIZATION,
    )
    incident_gap_raman = gaussian_incident_field(
        gap_pos3_m,
        np.r_[source_xy_nm, 0.0] * 1e-9,
        BEAM_WAIST_NM * 1e-9,
        INCIDENT_AMPLITUDE_V_PER_M,
        POLARIZATION,
    )
    print("Evaluating excitation- and Raman-wavelength fields at every physical gap center ...")
    e_gap_exc_cda = evaluate_total_field(
        gap_pos3_m, pos3_m, dipoles, incident_gap_exc, k,
        water_epsilon_exc,
    )
    e_gap_raman_cda = evaluate_total_field(
        gap_pos3_m, pos3_m, dipoles_raman, incident_gap_raman, k_raman,
        water_epsilon_raman,
    )
    c_exc, c_raman, calibration_status = calibration_factors(aggregate.edge_gaps_nm)
    e_gap_exc = e_gap_exc_cda * c_exc[:, None]
    e_gap_raman = e_gap_raman_cda * c_raman[:, None]

    e0_squared = abs(INCIDENT_AMPLITUDE_V_PER_M) ** 2
    gap_gain_exc = np.sum(np.abs(e_gap_exc) ** 2, axis=1) / e0_squared
    gap_gain_raman = np.sum(np.abs(e_gap_raman) ** 2, axis=1) / e0_squared
    gap_sers_ef = gap_gain_exc * gap_gain_raman
    gap_sers_e4_proxy = gap_gain_exc**2

    # Assign a graph/path coordinate to every gap using the validated BFS tree.
    node_tree_dg_nm = node_tree_geodesic_nm(
        aggregate, source, graph_distance, parent
    )
    gap_dg_nm = np.empty(len(aggregate.edges), dtype=float)
    for e, (i, j) in enumerate(aggregate.edges):
        edge_len = np.linalg.norm(aggregate.positions_nm[j] - aggregate.positions_nm[i])
        gap_dg_nm[e] = min(node_tree_dg_nm[i], node_tree_dg_nm[j]) + 0.5 * edge_len
    gap_euclidean_nm = np.linalg.norm(gap_xy_nm - source_xy_nm, axis=1)
    gap_incident_normalized = np.sum(np.abs(incident_gap_exc) ** 2, axis=1) / e0_squared
    remote_gap = (
        (gap_incident_normalized < REMOTE_INCIDENT_THRESHOLD)
        & (gap_euclidean_nm > REMOTE_DISTANCE_NM)
    )

    # Gap series along the exact particle path selected by the original code.
    edge_lookup = {
        tuple(sorted((int(i), int(j)))): e
        for e, (i, j) in enumerate(aggregate.edges)
    }
    path_gap_indices = np.asarray([
        edge_lookup[tuple(sorted((int(i), int(j))))]
        for i, j in zip(path[:-1], path[1:])
    ], dtype=int)
    path_gap_dg_nm = cumulative_geodesic_nm[:-1] + 0.5 * edge_lengths_nm
    path_gap_exc_relative = gap_gain_exc[path_gap_indices] / max(
        gap_gain_exc[path_gap_indices[0]], np.finfo(float).tiny
    )
    path_gap_sers_relative = gap_sers_ef[path_gap_indices] / max(
        gap_sers_ef[path_gap_indices[0]], np.finfo(float).tiny
    )
    path_gap_exc_envelope = smooth_monotone_log_envelope(
        path_gap_dg_nm, path_gap_exc_relative, SMOOTHING_WINDOW
    )
    path_gap_sers_envelope = smooth_monotone_log_envelope(
        path_gap_dg_nm, path_gap_sers_relative, SMOOTHING_WINDOW
    )
    gap_L_1e_nm = threshold_distance(path_gap_dg_nm, path_gap_exc_envelope, 1 / math.e)
    gap_L_1pct_nm = threshold_distance(path_gap_dg_nm, path_gap_exc_envelope, 1e-2)
    sers_L_1e_nm = threshold_distance(path_gap_dg_nm, path_gap_sers_envelope, 1 / math.e)
    sers_L_1pct_nm = threshold_distance(path_gap_dg_nm, path_gap_sers_envelope, 1e-2)
    gap_sers_ratios = path_gap_sers_relative[1:] / np.maximum(
        path_gap_sers_relative[:-1], np.finfo(float).tiny
    )
    gap_sers_up = np.flatnonzero(gap_sers_ratios > 1.0)

    remote_gap_indices = np.flatnonzero(remote_gap)
    if len(remote_gap_indices):
        strongest_remote_gap = int(
            remote_gap_indices[np.argmax(gap_sers_ef[remote_gap_indices])]
        )
        strongest_remote_gap_sers_ef = float(gap_sers_ef[strongest_remote_gap])
        strongest_remote_gap_dg_nm = float(gap_dg_nm[strongest_remote_gap])
        strongest_remote_gap_euclidean_nm = float(gap_euclidean_nm[strongest_remote_gap])
        strongest_remote_gap_relative_to_global = float(
            gap_sers_ef[strongest_remote_gap] / gap_sers_ef.max()
        )
    else:
        strongest_remote_gap = -1
        strongest_remote_gap_sers_ef = float("nan")
        strongest_remote_gap_dg_nm = float("nan")
        strongest_remote_gap_euclidean_nm = float("nan")
        strongest_remote_gap_relative_to_global = float("nan")

    # Relative-to-source transport metric used for L_1/e and L_1%.
    raw_path_global = dipole_normalized[path]
    raw_path_relative = dipole_intensity[path] / max(
        dipole_intensity[path[0]], np.finfo(float).tiny
    )
    envelope = smooth_monotone_log_envelope(
        cumulative_geodesic_nm,
        raw_path_relative,
        SMOOTHING_WINDOW,
    )

    L_1e_nm = threshold_distance(cumulative_geodesic_nm, envelope, 1 / math.e)
    L_1pct_nm = threshold_distance(cumulative_geodesic_nm, envelope, 1e-2)
    dG_max_nm = float(cumulative_geodesic_nm[-1])

    # Path-local downstream enhancement events.
    ratios = raw_path_relative[1:] / np.maximum(
        raw_path_relative[:-1], np.finfo(float).tiny
    )
    enhancement_steps = np.flatnonzero(ratios > 1.0)
    n_enh = int(len(enhancement_steps))
    enhancement_fraction = float(n_enh / len(ratios)) if len(ratios) else 0.0
    max_enhancement_ratio = float(ratios.max()) if len(ratios) else float("nan")
    max_enhancement_step = int(np.argmax(ratios)) if len(ratios) else -1
    max_enhancement_geodesic_nm = (
        float(cumulative_geodesic_nm[max_enhancement_step + 1])
        if max_enhancement_step >= 0
        else float("nan")
    )

    # Level 2.6 whole-network particle downstream scan (all geometry edges),
    # restored in addition to the selected-path statistics above.
    particle_downstream_rows = particle_downstream_analysis(
        aggregate, graph_distance, dipole_intensity, euclidean_nm
    )
    particle_event_rows = [
        row for row in particle_downstream_rows if row["is_enhancement"]
    ]
    particle_event_probability = (
        len(particle_event_rows) / len(particle_downstream_rows)
        if particle_downstream_rows else 0.0
    )
    # Correct Level-2.6 maximum: the original script accidentally used the
    # edge_in_loop column rather than enhancement_ratio for this one summary.
    strongest_particle_event = max(
        particle_event_rows,
        key=lambda row: row["enhancement_ratio"],
        default=None,
    )

    # Level 2.8 whole-network gap-SERS downstream scan on the line graph.
    gap_downstream_rows = gap_sers_downstream_analysis(
        aggregate, gap_dg_nm, gap_euclidean_nm, gap_sers_ef
    )
    gap_event_rows = [
        row for row in gap_downstream_rows if row["is_enhancement"]
    ]
    gap_event_probability = (
        len(gap_event_rows) / len(gap_downstream_rows)
        if gap_downstream_rows else 0.0
    )
    strongest_gap_event = max(
        gap_event_rows,
        key=lambda row: row["enhancement_ratio"],
        default=None,
    )

    # Geometry summary.
    x_min, y_min = aggregate.positions_nm.min(axis=0)
    x_max, y_max = aggregate.positions_nm.max(axis=0)
    width_nm = float(x_max - x_min)
    height_nm = float(y_max - y_min)
    bbox_area_nm2 = max(width_nm * height_nm, np.finfo(float).tiny)
    number_density_per_um2 = float(n / (bbox_area_nm2 * 1e-6))

    # Save particle/path data.
    write_csv(
        data_dir / "particles.csv",
        [
            "id", "x_nm", "y_nm", "radius_nm", "degree", "euclidean_nm",
            "graph_distance", "incident_normalized", "dipole_intensity",
            "dipole_normalized", "remote"
        ],
        (
            [
                i,
                aggregate.positions_nm[i, 0],
                aggregate.positions_nm[i, 1],
                aggregate.radii_nm[i],
                aggregate.degrees[i],
                euclidean_nm[i],
                graph_distance[i],
                incident_normalized[i],
                dipole_intensity[i],
                dipole_normalized[i],
                int(remote[i]),
            ]
            for i in range(n)
        ),
    )

    write_csv(
        data_dir / "edges.csv",
        ["i", "j", "surface_gap_nm"],
        ([int(i), int(j), g] for (i, j), g in zip(aggregate.edges, aggregate.edge_gaps_nm)),
    )

    write_csv(
        data_dir / "path.csv",
        [
            "step", "id", "x_nm", "y_nm", "euclidean_nm",
            "cumulative_geodesic_nm", "dipole_normalized_global",
            "M_relative_to_source", "smoothed_monotone_envelope"
        ],
        (
            [
                step,
                int(pid),
                aggregate.positions_nm[pid, 0],
                aggregate.positions_nm[pid, 1],
                euclidean_nm[pid],
                cumulative_geodesic_nm[step],
                raw_path_global[step],
                raw_path_relative[step],
                envelope[step],
            ]
            for step, pid in enumerate(path)
        ),
    )

    write_csv(
        data_dir / "path_particle_enhancement_events.csv",
        [
            "upstream_step", "downstream_step", "upstream_id", "downstream_id",
            "downstream_geodesic_nm", "enhancement_ratio"
        ],
        (
            [
                int(s),
                int(s + 1),
                int(path[s]),
                int(path[s + 1]),
                cumulative_geodesic_nm[s + 1],
                ratios[s],
            ]
            for s in enhancement_steps
        ),
    )

    particle_header = [
        "edge_id", "upstream_id", "downstream_id",
        "upstream_graph_distance", "downstream_graph_distance",
        "upstream_coordination", "downstream_coordination",
        "upstream_mean_local_gap_nm", "downstream_mean_local_gap_nm",
        "connecting_gap_nm", "edge_in_loop", "downstream_euclidean_nm",
        "enhancement_ratio", "is_enhancement",
    ]
    write_csv(
        data_dir / "particle_downstream_edges.csv",
        particle_header,
        ([row[key] for key in particle_header] for row in particle_downstream_rows),
    )
    write_csv(
        data_dir / "particle_downstream_enhancement_events.csv",
        particle_header,
        ([row[key] for key in particle_header] for row in particle_event_rows),
    )
    # Compatibility with the exact Level-2.6 output names.
    write_csv(
        data_dir / "downstream_edges.csv",
        particle_header,
        ([row[key] for key in particle_header] for row in particle_downstream_rows),
    )
    write_csv(
        data_dir / "enhancement_events.csv",
        particle_header,
        ([row[key] for key in particle_header] for row in particle_event_rows),
    )
    particle_coordination_rows = grouped_probability(
        particle_downstream_rows, "upstream_coordination"
    )
    write_csv(
        data_dir / "coordination_summary.csv",
        ["upstream_coordination", "downstream_edges", "enhancement_events",
         "enhancement_probability"],
        particle_coordination_rows,
    )

    gap_header = [
        "shared_particle", "shared_particle_coordination", "upstream_gap_id",
        "downstream_gap_id", "upstream_gap_geodesic_nm",
        "downstream_gap_geodesic_nm", "downstream_gap_euclidean_nm",
        "upstream_surface_gap_nm", "downstream_surface_gap_nm",
        "upstream_SERS_EF", "downstream_SERS_EF", "enhancement_ratio",
        "is_enhancement",
    ]
    write_csv(
        data_dir / "gap_SERS_downstream_pairs.csv",
        gap_header,
        ([row[key] for key in gap_header] for row in gap_downstream_rows),
    )
    write_csv(
        data_dir / "gap_SERS_downstream_enhancement_events.csv",
        gap_header,
        ([row[key] for key in gap_header] for row in gap_event_rows),
    )
    gap_coordination_rows = grouped_probability(
        gap_downstream_rows, "shared_particle_coordination"
    )
    write_csv(
        data_dir / "gap_SERS_coordination_summary.csv",
        ["shared_particle_coordination", "downstream_gap_pairs",
         "enhancement_events", "enhancement_probability"],
        gap_coordination_rows,
    )

    # Full hotspot/gap table: one row for every physical near-contact edge.
    write_csv(
        data_dir / "gaps_Efield_SERS.csv",
        [
            "gap_id", "i", "j", "x_gap_nm", "y_gap_nm", "surface_gap_nm",
            "gap_geodesic_nm", "gap_euclidean_nm", "incident_normalized",
            "remote_gap", "C_exc_abs", "C_raman_abs",
            "Eexc_x_real", "Eexc_x_imag", "Eexc_y_real", "Eexc_y_imag",
            "Eexc_z_real", "Eexc_z_imag", "Eraman_x_real", "Eraman_x_imag",
            "Eraman_y_real", "Eraman_y_imag", "Eraman_z_real", "Eraman_z_imag",
            "field_intensity_gain_exc", "field_intensity_gain_raman",
            "SERS_EF_two_frequency", "SERS_E4_same_frequency_proxy",
            "SERS_normalized_global",
        ],
        (
            [
                e, int(i), int(j), gap_xy_nm[e, 0], gap_xy_nm[e, 1],
                aggregate.edge_gaps_nm[e], gap_dg_nm[e], gap_euclidean_nm[e],
                gap_incident_normalized[e], int(remote_gap[e]), c_exc[e], c_raman[e],
                e_gap_exc[e, 0].real, e_gap_exc[e, 0].imag,
                e_gap_exc[e, 1].real, e_gap_exc[e, 1].imag,
                e_gap_exc[e, 2].real, e_gap_exc[e, 2].imag,
                e_gap_raman[e, 0].real, e_gap_raman[e, 0].imag,
                e_gap_raman[e, 1].real, e_gap_raman[e, 1].imag,
                e_gap_raman[e, 2].real, e_gap_raman[e, 2].imag,
                gap_gain_exc[e], gap_gain_raman[e], gap_sers_ef[e],
                gap_sers_e4_proxy[e], gap_sers_ef[e] / gap_sers_ef.max(),
            ]
            for e, (i, j) in enumerate(aggregate.edges)
        ),
    )

    write_csv(
        data_dir / "path_gaps.csv",
        [
            "path_gap_step", "gap_id", "i", "j", "gap_geodesic_nm",
            "surface_gap_nm", "field_gain_exc", "field_gain_raman",
            "SERS_EF", "gap_exc_relative_to_first", "gap_SERS_relative_to_first",
            "gap_exc_envelope", "gap_SERS_envelope",
        ],
        (
            [
                s, int(e), int(aggregate.edges[e, 0]), int(aggregate.edges[e, 1]),
                path_gap_dg_nm[s], aggregate.edge_gaps_nm[e], gap_gain_exc[e],
                gap_gain_raman[e], gap_sers_ef[e], path_gap_exc_relative[s],
                path_gap_sers_relative[s], path_gap_exc_envelope[s],
                path_gap_sers_envelope[s],
            ]
            for s, e in enumerate(path_gap_indices)
        ),
    )

    write_csv(
        data_dir / "remote_gaps.csv",
        [
            "gap_id", "i", "j", "euclidean_nm", "geodesic_nm",
            "surface_gap_nm", "field_gain_exc", "field_gain_raman", "SERS_EF",
            "SERS_normalized_global",
        ],
        (
            [
                int(e), int(aggregate.edges[e, 0]), int(aggregate.edges[e, 1]),
                gap_euclidean_nm[e], gap_dg_nm[e], aggregate.edge_gaps_nm[e],
                gap_gain_exc[e], gap_gain_raman[e], gap_sers_ef[e],
                gap_sers_ef[e] / gap_sers_ef.max(),
            ]
            for e in remote_gap_indices
        ),
    )

    # This template prevents an undocumented calibration factor being invented.
    write_csv(
        data_dir / "level3B_calibration_template.csv",
        ["gap_nm", "C_exc_abs", "C_raman_abs"],
        [],
    )

    # Network figure.
    segments = np.array(
        [[aggregate.positions_nm[i], aggregate.positions_nm[j]] for i, j in aggregate.edges]
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.add_collection(LineCollection(segments, colors="0.85", linewidths=0.7))
    v = np.maximum(dipole_normalized, 1e-12)
    sc = ax.scatter(
        aggregate.positions_nm[:, 0],
        aggregate.positions_nm[:, 1],
        c=v,
        s=28,
        cmap="viridis",
        norm=LogNorm(vmin=1e-12, vmax=1.0),
        zorder=2,
    )
    ax.plot(
        aggregate.positions_nm[path, 0],
        aggregate.positions_nm[path, 1],
        "o-",
        markersize=3,
        linewidth=1.2,
        label=f"selected path: {dG_max_nm:.1f} nm",
        zorder=3,
    )
    ax.scatter(*source_xy_nm, marker="*", s=110, label="source", zorder=4)
    fig.colorbar(sc, ax=ax, label="normalized dipole intensity")
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"2D coupled-dipole network — N={n}, seed={seed}")
    ax.legend(loc="best")
    save_figure(fig, out_dir / "network_and_remote_path")

    # Restored Level-2.6 whole-network particle downstream enhancement map.
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.add_collection(LineCollection(segments, colors="0.84", linewidths=0.8))
    if particle_event_rows:
        event_segments = np.array([
            [aggregate.positions_nm[row["upstream_id"]],
             aggregate.positions_nm[row["downstream_id"]]]
            for row in particle_event_rows
        ])
        event_ratios = np.array([
            row["enhancement_ratio"] for row in particle_event_rows
        ])
        event_collection = LineCollection(
            event_segments,
            cmap="Reds",
            norm=LogNorm(vmin=1.0, vmax=max(1.000001, float(event_ratios.max()))),
            linewidths=2.4,
            zorder=3,
        )
        event_collection.set_array(event_ratios)
        ax.add_collection(event_collection)
        fig.colorbar(event_collection, ax=ax, label="downstream dipole-intensity ratio")
    ax.scatter(aggregate.positions_nm[:, 0], aggregate.positions_nm[:, 1],
               s=9, color="0.35", zorder=2)
    ax.scatter(*source_xy_nm, marker="*", s=125, color="gold", edgecolor="black",
               label="source", zorder=4)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(
        f"Particle downstream enhancement map — N={n}, seed={seed}\n"
        f"{len(particle_event_rows)}/{len(particle_downstream_rows)} directed edges"
    )
    ax.legend(loc="best")
    save_figure(fig, out_dir / "particle_downstream_enhancement_map")

    # Original Level-2.6 probability versus upstream coordination.
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    coordination = np.array([row[0] for row in particle_coordination_rows])
    coordination_probability = np.array([row[3] for row in particle_coordination_rows])
    ax.bar(coordination, coordination_probability, color="#4c78a8")
    ax.set_xticks(coordination)
    ax.set_ylim(0, max(0.05, 1.12 * float(coordination_probability.max())))
    ax.set_xlabel("upstream particle coordination number")
    ax.set_ylabel("P(downstream dipole enhancement)")
    ax.set_title(f"Particle enhancement vs coordination — N={n}, seed={seed}")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, out_dir / "enhancement_vs_coordination")

    # Particle enhancement ratio and binned probability versus connecting gap.
    particle_gaps = np.array([
        row["connecting_gap_nm"] for row in particle_downstream_rows
    ])
    particle_ratios = np.array([
        row["enhancement_ratio"] for row in particle_downstream_rows
    ])
    particle_is_event = np.array([
        bool(row["is_enhancement"]) for row in particle_downstream_rows
    ])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax0.scatter(particle_gaps[~particle_is_event], particle_ratios[~particle_is_event],
                s=15, color="0.65", alpha=0.55, label="ratio ≤ 1")
    ax0.scatter(particle_gaps[particle_is_event], particle_ratios[particle_is_event],
                s=22, color="#d62728", alpha=0.8, label="enhancement")
    ax0.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax0.set_yscale("log")
    ax0.set_xlabel("connecting surface gap (nm)")
    ax0.set_ylabel("downstream/upstream dipole intensity")
    ax0.legend(loc="best")
    gap_bins = np.linspace(float(particle_gaps.min()), float(particle_gaps.max()), 7)
    centers, probabilities = [], []
    for lo, hi in zip(gap_bins[:-1], gap_bins[1:]):
        mask = (particle_gaps >= lo) & (
            particle_gaps <= hi if hi == gap_bins[-1] else particle_gaps < hi
        )
        if np.any(mask):
            centers.append(0.5 * (lo + hi))
            probabilities.append(float(np.mean(particle_is_event[mask])))
    ax1.plot(centers, probabilities, "o-", color="#d62728")
    ax1.set_ylim(0, max(0.05, 1.12 * max(probabilities)))
    ax1.set_xlabel("connecting surface gap bin (nm)")
    ax1.set_ylabel("enhancement probability")
    ax1.grid(True, alpha=0.25)
    fig.suptitle(f"Particle downstream enhancement vs gap — N={n}, seed={seed}")
    save_figure(fig, out_dir / "enhancement_vs_gap")

    # Particle enhancement on loop edges versus bridge/nonloop edges.
    loop_rows = [row for row in particle_downstream_rows if row["edge_in_loop"]]
    nonloop_rows = [row for row in particle_downstream_rows if not row["edge_in_loop"]]
    categories = ["nonloop / bridge", "loop edge"]
    category_rows = [nonloop_rows, loop_rows]
    loop_probabilities = [
        sum(row["is_enhancement"] for row in rows) / len(rows) if rows else 0.0
        for rows in category_rows
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    bars = ax.bar(categories, loop_probabilities, color=["#9c755f", "#59a14f"])
    for bar, rows in zip(bars, category_rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{sum(row['is_enhancement'] for row in rows)}/{len(rows)}",
                ha="center", va="bottom")
    ax.set_ylim(0, max(0.05, 1.18 * max(loop_probabilities)))
    ax.set_ylabel("P(downstream dipole enhancement)")
    ax.set_title(f"Loop vs nonloop particle enhancement — N={n}, seed={seed}")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, out_dir / "loop_vs_nonloop_enhancement")

    # Transport figure.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(
        cumulative_geodesic_nm,
        raw_path_relative,
        "o-",
        markersize=4,
        label="raw M(dG) / M(source)",
    )
    ax.semilogy(
        cumulative_geodesic_nm,
        envelope,
        "-",
        linewidth=2,
        label="smoothed monotone envelope",
    )
    ax.axhline(1 / math.e, linestyle="--", linewidth=1, label="1/e")
    ax.axhline(1e-2, linestyle=":", linewidth=1, label="1%")
    if np.isfinite(L_1e_nm):
        ax.axvline(L_1e_nm, linestyle="--", linewidth=1)
    if np.isfinite(L_1pct_nm):
        ax.axvline(L_1pct_nm, linestyle=":", linewidth=1)
    ax.set_xlabel("cumulative geodesic distance (nm)")
    ax.set_ylabel("M / M(source)")
    ax.set_title(f"Remote transport path — N={n}, seed={seed}")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.25)
    save_figure(fig, out_dir / "remote_transport_path")

    # Gap-hotspot network map: edges, not particles, carry the SERS observable.
    fig, ax = plt.subplots(figsize=(9, 7))
    sers_plot = np.maximum(gap_sers_ef, np.finfo(float).tiny)
    edge_collection = LineCollection(
        segments,
        cmap="magma",
        norm=LogNorm(vmin=max(float(sers_plot.min()), 1e-12), vmax=float(sers_plot.max())),
        linewidths=2.0,
    )
    edge_collection.set_array(sers_plot)
    ax.add_collection(edge_collection)
    ax.scatter(
        aggregate.positions_nm[:, 0], aggregate.positions_nm[:, 1],
        s=7, color="0.6", zorder=2,
    )
    ax.scatter(*source_xy_nm, marker="*", s=120, color="cyan", edgecolor="black",
               label="source", zorder=4)
    fig.colorbar(edge_collection, ax=ax, label="two-frequency CDA SERS EF proxy")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"Gap-hotspot distribution — N={n}, seed={seed}")
    ax.legend(loc="best")
    save_figure(fig, out_dir / "gap_hotspot_SERS_map")

    # Level-2.8 whole-network gap-SERS downstream enhancement map.  Red/orange
    # edges are downstream gaps that are stronger than an adjacent upstream gap.
    downstream_gap_max_ratio = {}
    for row in gap_event_rows:
        gap_id = row["downstream_gap_id"]
        downstream_gap_max_ratio[gap_id] = max(
            downstream_gap_max_ratio.get(gap_id, 1.0), row["enhancement_ratio"]
        )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.add_collection(LineCollection(segments, colors="0.86", linewidths=0.9))
    if downstream_gap_max_ratio:
        marked_gap_ids = np.array(sorted(downstream_gap_max_ratio), dtype=int)
        marked_ratios = np.array([
            downstream_gap_max_ratio[e] for e in marked_gap_ids
        ])
        marked_collection = LineCollection(
            segments[marked_gap_ids],
            cmap="autumn_r",
            norm=LogNorm(vmin=1.0, vmax=max(1.000001, float(marked_ratios.max()))),
            linewidths=3.0,
            zorder=3,
        )
        marked_collection.set_array(marked_ratios)
        ax.add_collection(marked_collection)
        fig.colorbar(marked_collection, ax=ax,
                     label="maximum adjacent downstream gap-SERS ratio")
    ax.scatter(aggregate.positions_nm[:, 0], aggregate.positions_nm[:, 1],
               s=8, color="0.4", zorder=2)
    ax.scatter(*source_xy_nm, marker="*", s=125, color="cyan", edgecolor="black",
               label="source", zorder=4)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(
        f"Gap-SERS downstream enhancement map — N={n}, seed={seed}\n"
        f"{len(gap_event_rows)}/{len(gap_downstream_rows)} adjacent directed gap pairs"
    )
    ax.legend(loc="best")
    save_figure(fig, out_dir / "gap_SERS_downstream_enhancement_map")

    # Gap-SERS downstream ratio versus graph/geodesic distance.
    gap_pair_distances = np.array([
        row["downstream_gap_geodesic_nm"] for row in gap_downstream_rows
    ])
    gap_pair_ratios = np.array([
        row["enhancement_ratio"] for row in gap_downstream_rows
    ])
    gap_pair_is_event = np.array([
        bool(row["is_enhancement"]) for row in gap_downstream_rows
    ])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(gap_pair_distances[~gap_pair_is_event],
               gap_pair_ratios[~gap_pair_is_event], s=13, color="0.68",
               alpha=0.45, label="ratio ≤ 1")
    ax.scatter(gap_pair_distances[gap_pair_is_event],
               gap_pair_ratios[gap_pair_is_event], s=20, color="#e66101",
               alpha=0.75, label="gap-SERS enhancement")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("downstream gap graph/geodesic coordinate (nm)")
    ax.set_ylabel("downstream/upstream two-frequency SERS EF")
    ax.set_title(f"Gap-SERS enhancement vs distance — N={n}, seed={seed}")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best")
    save_figure(fig, out_dir / "gap_SERS_enhancement_vs_distance")

    # Gap-SERS enhancement probability versus shared-particle coordination.
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    gap_coordination = np.array([row[0] for row in gap_coordination_rows])
    gap_coordination_probability = np.array([row[3] for row in gap_coordination_rows])
    ax.bar(gap_coordination, gap_coordination_probability, color="#f28e2b")
    ax.set_xticks(gap_coordination)
    ax.set_ylim(0, max(0.05, 1.12 * float(gap_coordination_probability.max())))
    ax.set_xlabel("coordination of particle shared by adjacent gaps")
    ax.set_ylabel("P(downstream gap-SERS enhancement)")
    ax.set_title(f"Gap-SERS enhancement vs coordination — N={n}, seed={seed}")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, out_dir / "gap_SERS_enhancement_vs_coordination")

    # Selected-path gap fields: the direct bridge from particle transport to SERS.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(
        path_gap_dg_nm, path_gap_exc_relative, "o-", markersize=3,
        label=rf"gap $|E({WAVELENGTH_NM:.0f}\,\mathrm{{nm}})|^2$ / first gap",
    )
    ax.semilogy(
        path_gap_dg_nm, path_gap_sers_relative, "s-", markersize=3,
        label="two-frequency SERS EF / first gap",
    )
    ax.semilogy(
        path_gap_dg_nm, path_gap_sers_envelope, "k--", linewidth=1.7,
        label="smoothed SERS envelope",
    )
    ax.axhline(1 / math.e, linestyle="--", color="0.5", linewidth=1)
    ax.axhline(1e-2, linestyle=":", color="0.3", linewidth=1)
    ax.set_xlabel("cumulative path distance to gap center (nm)")
    ax.set_ylabel("relative hotspot / SERS observable")
    ax.set_title(f"Gap-hotspot propagation — N={n}, seed={seed}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig, out_dir / "gap_SERS_along_selected_path")

    # All gaps versus graph coordinate, explicitly retaining local re-concentration.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(
        gap_dg_nm, sers_plot, c=aggregate.edge_gaps_nm, s=17, alpha=0.72,
        cmap="viridis_r",
    )
    ax.set_yscale("log")
    ax.set_xlabel("gap graph/geodesic coordinate (nm)")
    ax.set_ylabel("two-frequency CDA SERS EF proxy")
    ax.set_title(f"All gap hotspots versus network distance — N={n}, seed={seed}")
    fig.colorbar(scatter, ax=ax, label="surface gap (nm)")
    ax.grid(True, which="both", alpha=0.2)
    save_figure(fig, out_dir / "all_gaps_SERS_vs_distance")

    elapsed_s = time.perf_counter() - t0

    summary = {
        "N": int(n),
        "seed": int(seed),
        "excitation_wavelength_nm": float(WAVELENGTH_NM),
        "silver_epsilon_exc_real": float(silver_epsilon_exc.real),
        "silver_epsilon_exc_imag": float(silver_epsilon_exc.imag),
        "water_refractive_index_exc": float(water_n_exc),
        "water_relative_permittivity_exc_real": float(water_epsilon_exc.real),
        "water_relative_permittivity_exc_imag": float(water_epsilon_exc.imag),
        "width_nm": width_nm,
        "height_nm": height_nm,
        "bbox_number_density_per_um2": number_density_per_um2,
        "mean_gap_nm": float(np.mean(aggregate.edge_gaps_nm)),
        "median_gap_nm": float(np.median(aggregate.edge_gaps_nm)),
        "std_gap_nm": float(np.std(aggregate.edge_gaps_nm)),
        "min_all_pair_surface_gap_nm": float(minimum_surface_gap(aggregate)),
        "mean_coordination": float(np.mean(aggregate.degrees)),
        "max_coordination": int(np.max(aggregate.degrees)),
        "source_particle": int(source),
        "selected_target_particle": int(target),
        "target_euclidean_nm": float(euclidean_nm[target]),
        "maximum_hop_number": int(len(path) - 1),
        "dG_max_nm": dG_max_nm,
        "L_1e_nm": L_1e_nm,
        "L_1pct_nm": L_1pct_nm,
        "path_enhancement_events": n_enh,
        "path_enhancement_fraction": enhancement_fraction,
        "max_path_enhancement_ratio": max_enhancement_ratio,
        "max_path_enhancement_geodesic_nm": max_enhancement_geodesic_nm,
        "particle_downstream_edges": int(len(particle_downstream_rows)),
        "particle_downstream_enhancement_events": int(len(particle_event_rows)),
        "particle_downstream_enhancement_probability": float(
            particle_event_probability
        ),
        "maximum_particle_downstream_enhancement_ratio": float(
            strongest_particle_event["enhancement_ratio"]
            if strongest_particle_event else float("nan")
        ),
        "maximum_particle_downstream_upstream_id": int(
            strongest_particle_event["upstream_id"]
            if strongest_particle_event else -1
        ),
        "maximum_particle_downstream_downstream_id": int(
            strongest_particle_event["downstream_id"]
            if strongest_particle_event else -1
        ),
        "cda_relative_residual": residual,
        "raman_shift_cm1": float(RAMAN_SHIFT_CM1),
        "raman_wavelength_nm": float(wavelength_raman_nm),
        "silver_epsilon_raman_real": float(silver_epsilon_raman.real),
        "silver_epsilon_raman_imag": float(silver_epsilon_raman.imag),
        "water_refractive_index_raman": float(water_n_raman),
        "water_relative_permittivity_raman_real": float(water_epsilon_raman.real),
        "water_relative_permittivity_raman_imag": float(water_epsilon_raman.imag),
        "silver_permittivity_model": "linear interpolation of Ag_Johnson_eps.csv",
        "water_dispersion_model": "Daimon-Masumura 2007 at 19 C, real n",
        "cda_relative_residual_raman": residual_raman,
        "local_field_calibration_status": calibration_status,
        "number_of_gaps": int(len(aggregate.edges)),
        "number_of_remote_gaps": int(len(remote_gap_indices)),
        "gap_field_L_1e_nm": gap_L_1e_nm,
        "gap_field_L_1pct_nm": gap_L_1pct_nm,
        "gap_SERS_L_1e_nm": sers_L_1e_nm,
        "gap_SERS_L_1pct_nm": sers_L_1pct_nm,
        "maximum_gap_field_gain_exc": float(gap_gain_exc.max()),
        "maximum_gap_field_gain_raman": float(gap_gain_raman.max()),
        "maximum_gap_SERS_EF": float(gap_sers_ef.max()),
        "median_gap_SERS_EF": float(np.median(gap_sers_ef)),
        "path_gap_SERS_enhancement_events": int(len(gap_sers_up)),
        "path_gap_SERS_enhancement_fraction": float(
            len(gap_sers_up) / max(1, len(gap_sers_ratios))
        ),
        "gap_SERS_downstream_pairs": int(len(gap_downstream_rows)),
        "gap_SERS_downstream_enhancement_events": int(len(gap_event_rows)),
        "gap_SERS_downstream_enhancement_probability": float(
            gap_event_probability
        ),
        "maximum_gap_SERS_downstream_enhancement_ratio": float(
            strongest_gap_event["enhancement_ratio"]
            if strongest_gap_event else float("nan")
        ),
        "maximum_gap_SERS_downstream_upstream_gap_id": int(
            strongest_gap_event["upstream_gap_id"]
            if strongest_gap_event else -1
        ),
        "maximum_gap_SERS_downstream_downstream_gap_id": int(
            strongest_gap_event["downstream_gap_id"]
            if strongest_gap_event else -1
        ),
        "strongest_remote_gap_id": strongest_remote_gap,
        "strongest_remote_gap_SERS_EF": strongest_remote_gap_sers_ef,
        "strongest_remote_gap_dG_nm": strongest_remote_gap_dg_nm,
        "strongest_remote_gap_euclidean_nm": strongest_remote_gap_euclidean_nm,
        "strongest_remote_gap_SERS_relative_global": strongest_remote_gap_relative_to_global,
        "elapsed_seconds": elapsed_s,
    }

    write_csv(
        data_dir / "summary.csv",
        ["metric", "value"],
        summary.items(),
    )

    print(
        f"Finished N={n}, seed={seed}: dGmax={dG_max_nm:.1f} nm, "
        f"L1/e={L_1e_nm:.1f} nm, L1%={L_1pct_nm:.1f} nm, "
        f"gap-SERS L1/e={sers_L_1e_nm:.1f} nm, "
        f"particle events={len(particle_event_rows)}/{len(particle_downstream_rows)}, "
        f"gap-SERS events={len(gap_event_rows)}/{len(gap_downstream_rows)}, "
        f"remote gaps={len(remote_gap_indices)}, time={elapsed_s:.1f} s"
    )

    gap_payload = (
        path_gap_dg_nm,
        path_gap_sers_relative,
        path_gap_sers_envelope,
    )
    return summary, cumulative_geodesic_nm, raw_path_relative, envelope, gap_payload


# =============================================================================
# 9. ENSEMBLE AND SIZE-CONVERGENCE DRIVER
# =============================================================================
def interpolate_curve(distance_nm, values, grid_nm):
    values = np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)
    return 10 ** np.interp(grid_nm, distance_nm, np.log10(values))


def finite_mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def run_study():
    if RUN_MODE == "baseline":
        sizes = [200]
        seeds = [BASE_SEED]
    elif RUN_MODE == "pilot":
        sizes = NETWORK_SIZES
        seeds = [BASE_SEED]
    elif RUN_MODE == "full":
        sizes = NETWORK_SIZES
        seeds = [BASE_SEED + i for i in range(FULL_NUMBER_OF_SEEDS)]
    else:
        raise ValueError("RUN_MODE must be 'baseline', 'pilot', or 'full'")

    run_root = OUTPUT_ROOT / RUN_MODE
    run_root.mkdir(parents=True, exist_ok=True)

    optics = optical_parameters()
    ag_table_sha256 = hashlib.sha256(AG_DIELECTRIC_CSV.read_bytes()).hexdigest()
    optical_rows = [
        ("excitation_wavelength_nm", optics["excitation_wavelength_nm"]),
        ("raman_shift_cm1", optics["raman_shift_cm1"]),
        ("raman_wavelength_nm", optics["raman_wavelength_nm"]),
        ("silver_epsilon_exc_real", optics["silver_epsilon_exc"].real),
        ("silver_epsilon_exc_imag", optics["silver_epsilon_exc"].imag),
        ("silver_epsilon_raman_real", optics["silver_epsilon_raman"].real),
        ("silver_epsilon_raman_imag", optics["silver_epsilon_raman"].imag),
        ("water_refractive_index_exc", optics["water_n_exc"]),
        ("water_refractive_index_raman", optics["water_n_raman"]),
        ("water_relative_permittivity_exc", optics["water_epsilon_exc"].real),
        ("water_relative_permittivity_raman", optics["water_epsilon_raman"].real),
        ("silver_table_filename", AG_DIELECTRIC_CSV.name),
        ("silver_table_sha256", ag_table_sha256),
        ("silver_table_interpolation", "linear in eps_real and eps_imag; no extrapolation"),
        ("water_model", "Daimon-Masumura 2007, 19 C, real refractive index"),
        ("water_absorption", "not included in this Level-2 model"),
    ]
    write_csv(run_root / "optical_parameters.csv", ["parameter", "value"], optical_rows)

    print("=" * 72)
    print("Remote-SERS Level 2.6–2.8: Ag in water, 785 nm excitation")
    print(f"RUN_MODE: {RUN_MODE}")
    print(f"Network sizes: {sizes}")
    print(f"Seeds: {seeds}")
    print(
        "Optics (excitation): "
        f"lambda={WAVELENGTH_NM:.3f} nm, "
        f"eps_Ag={optics['silver_epsilon_exc'].real:.6f}"
        f"+{optics['silver_epsilon_exc'].imag:.6f}j, "
        f"n_water={optics['water_n_exc']:.9f}"
    )
    print(
        "Optics (Stokes): "
        f"lambda={optics['raman_wavelength_nm']:.6f} nm, "
        f"eps_Ag={optics['silver_epsilon_raman'].real:.6f}"
        f"+{optics['silver_epsilon_raman'].imag:.6f}j, "
        f"n_water={optics['water_n_raman']:.9f}"
    )
    print(f"Output: {run_root}")
    print("=" * 72)

    all_rows = []
    curves_by_n = {n: [] for n in sizes}
    gap_curves_by_n = {n: [] for n in sizes}

    for n in sizes:
        for seed in seeds:
            realization_dir = run_root / f"N{n}" / f"seed_{seed}"
            try:
                summary, d, raw, env, gap_payload = run_one_realization(
                    n, seed, realization_dir
                )
                all_rows.append(summary)
                curves_by_n[n].append((d, raw, env))
                gap_curves_by_n[n].append(gap_payload)
            except Exception as exc:
                warnings.warn(f"N={n}, seed={seed} failed: {exc}")
                all_rows.append({
                    "N": n,
                    "seed": seed,
                    "error": repr(exc),
                })

    # Save realization-level summary.
    keys = sorted({k for row in all_rows for k in row.keys()})
    write_csv(
        run_root / "size_convergence_summary.csv",
        keys,
        ([row.get(k, "") for k in keys] for row in all_rows),
    )

    # Per-N ensemble M(dG).
    ensemble_rows = []
    for n, curves in curves_by_n.items():
        if not curves:
            continue

        common_max = min(float(d[-1]) for d, _, _ in curves)
        grid = np.linspace(0.0, common_max, 180)
        interpolated = np.vstack([
            interpolate_curve(d, raw, grid) for d, raw, _ in curves
        ])
        mean_m = interpolated.mean(axis=0)
        std_m = interpolated.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_m)

        ensemble_dir = run_root / f"N{n}" / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            ensemble_dir / "ensemble_M_vs_dG.csv",
            ["dG_nm", "mean_M_relative", "std_M_relative", "n_realizations"],
            ([x, m, s, len(curves)] for x, m, s in zip(grid, mean_m, std_m)),
        )

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.semilogy(grid, mean_m, linewidth=2, label=f"N={n}, ensemble mean")
        if len(curves) > 1:
            lower = np.maximum(mean_m - std_m, np.finfo(float).tiny)
            upper = mean_m + std_m
            ax.fill_between(grid, lower, upper, alpha=0.2, label="±1 SD")
        ax.set_xlabel("cumulative geodesic distance (nm)")
        ax.set_ylabel("ensemble M / M(source)")
        ax.set_title(f"Ensemble transport — N={n}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        save_figure(fig, ensemble_dir / "ensemble_transport")

        successful_rows = [r for r in all_rows if r.get("N") == n and "error" not in r]
        l1e_mean, l1e_std = finite_mean_std([r["L_1e_nm"] for r in successful_rows])
        l1pct_mean, l1pct_std = finite_mean_std([r["L_1pct_nm"] for r in successful_rows])
        dg_mean, dg_std = finite_mean_std([r["dG_max_nm"] for r in successful_rows])
        enh_mean, enh_std = finite_mean_std([r["path_enhancement_fraction"] for r in successful_rows])
        gap_l1e_mean, gap_l1e_std = finite_mean_std(
            [r["gap_field_L_1e_nm"] for r in successful_rows]
        )
        gap_l1pct_mean, gap_l1pct_std = finite_mean_std(
            [r["gap_field_L_1pct_nm"] for r in successful_rows]
        )
        sers_l1e_mean, sers_l1e_std = finite_mean_std(
            [r["gap_SERS_L_1e_nm"] for r in successful_rows]
        )
        sers_l1pct_mean, sers_l1pct_std = finite_mean_std(
            [r["gap_SERS_L_1pct_nm"] for r in successful_rows]
        )
        remote_sers_mean, remote_sers_std = finite_mean_std(
            [r["strongest_remote_gap_SERS_EF"] for r in successful_rows]
        )
        remote_relative_mean, remote_relative_std = finite_mean_std(
            [r["strongest_remote_gap_SERS_relative_global"] for r in successful_rows]
        )
        gap_up_mean, gap_up_std = finite_mean_std(
            [r["path_gap_SERS_enhancement_fraction"] for r in successful_rows]
        )
        particle_network_enh_mean, particle_network_enh_std = finite_mean_std(
            [r["particle_downstream_enhancement_probability"]
             for r in successful_rows]
        )
        gap_network_enh_mean, gap_network_enh_std = finite_mean_std(
            [r["gap_SERS_downstream_enhancement_probability"]
             for r in successful_rows]
        )

        ensemble_rows.append({
            "N": n,
            "successful_realizations": len(successful_rows),
            "dG_max_mean_nm": dg_mean,
            "dG_max_std_nm": dg_std,
            "L_1e_mean_nm": l1e_mean,
            "L_1e_std_nm": l1e_std,
            "L_1pct_mean_nm": l1pct_mean,
            "L_1pct_std_nm": l1pct_std,
            "path_enhancement_fraction_mean": enh_mean,
            "path_enhancement_fraction_std": enh_std,
            "gap_field_L_1e_mean_nm": gap_l1e_mean,
            "gap_field_L_1e_std_nm": gap_l1e_std,
            "gap_field_L_1pct_mean_nm": gap_l1pct_mean,
            "gap_field_L_1pct_std_nm": gap_l1pct_std,
            "gap_SERS_L_1e_mean_nm": sers_l1e_mean,
            "gap_SERS_L_1e_std_nm": sers_l1e_std,
            "gap_SERS_L_1pct_mean_nm": sers_l1pct_mean,
            "gap_SERS_L_1pct_std_nm": sers_l1pct_std,
            "strongest_remote_gap_SERS_EF_mean": remote_sers_mean,
            "strongest_remote_gap_SERS_EF_std": remote_sers_std,
            "strongest_remote_gap_SERS_relative_global_mean": remote_relative_mean,
            "strongest_remote_gap_SERS_relative_global_std": remote_relative_std,
            "gap_SERS_enhancement_fraction_mean": gap_up_mean,
            "gap_SERS_enhancement_fraction_std": gap_up_std,
            "particle_downstream_enhancement_probability_mean":
                particle_network_enh_mean,
            "particle_downstream_enhancement_probability_std":
                particle_network_enh_std,
            "gap_SERS_downstream_enhancement_probability_mean":
                gap_network_enh_mean,
            "gap_SERS_downstream_enhancement_probability_std":
                gap_network_enh_std,
        })

        # Ensemble SERS observable along the selected remote path.
        gap_curves = gap_curves_by_n[n]
        if gap_curves:
            common_gap_max = min(float(dg[-1]) for dg, _, _ in gap_curves)
            gap_grid = np.linspace(
                max(float(dg[0]) for dg, _, _ in gap_curves),
                common_gap_max,
                180,
            )
            gap_interp = np.vstack([
                interpolate_curve(dg, sers, gap_grid)
                for dg, sers, _ in gap_curves
            ])
            gap_mean = gap_interp.mean(axis=0)
            gap_std = (
                gap_interp.std(axis=0, ddof=1)
                if len(gap_curves) > 1 else np.zeros_like(gap_mean)
            )
            write_csv(
                ensemble_dir / "ensemble_gap_SERS_vs_dG.csv",
                ["dG_nm", "mean_SERS_relative", "std_SERS_relative", "n_realizations"],
                ([x, m, s, len(gap_curves)] for x, m, s in zip(gap_grid, gap_mean, gap_std)),
            )
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.semilogy(gap_grid, gap_mean, linewidth=2, label=f"N={n}, mean gap-SERS")
            if len(gap_curves) > 1:
                ax.fill_between(
                    gap_grid,
                    np.maximum(gap_mean-gap_std, np.finfo(float).tiny),
                    gap_mean+gap_std,
                    alpha=0.2,
                    label="±1 SD",
                )
            ax.set_xlabel("selected-path gap coordinate (nm)")
            ax.set_ylabel("SERS EF relative to first gap")
            ax.set_title(f"Ensemble gap-SERS transport — N={n}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
            save_figure(fig, ensemble_dir / "ensemble_gap_SERS_transport")

    if ensemble_rows:
        ekeys = list(ensemble_rows[0].keys())
        write_csv(
            run_root / "ensemble_size_summary.csv",
            ekeys,
            ([row[k] for k in ekeys] for row in ensemble_rows),
        )

    # Size-convergence plots if >= 2 sizes succeeded.
    if len(ensemble_rows) >= 2:
        Ns = np.array([r["N"] for r in ensemble_rows])

        for value_key, err_key, ylabel, filename in [
            ("L_1e_mean_nm", "L_1e_std_nm", "L1/e (nm)", "convergence_L1e"),
            ("L_1pct_mean_nm", "L_1pct_std_nm", "L1% (nm)", "convergence_L1pct"),
            ("dG_max_mean_nm", "dG_max_std_nm", "maximum tracked dG (nm)", "convergence_dGmax"),
            ("gap_field_L_1e_mean_nm", "gap_field_L_1e_std_nm", "gap-field L1/e (nm)", "convergence_gap_field_L1e"),
            ("gap_field_L_1pct_mean_nm", "gap_field_L_1pct_std_nm", "gap-field L1% (nm)", "convergence_gap_field_L1pct"),
            ("gap_SERS_L_1e_mean_nm", "gap_SERS_L_1e_std_nm", "gap-SERS L1/e (nm)", "convergence_gap_SERS_L1e"),
            ("gap_SERS_L_1pct_mean_nm", "gap_SERS_L_1pct_std_nm", "gap-SERS L1% (nm)", "convergence_gap_SERS_L1pct"),
        ]:
            y = np.array([r[value_key] for r in ensemble_rows], dtype=float)
            yerr = np.array([r[err_key] for r in ensemble_rows], dtype=float)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.errorbar(Ns, y, yerr=yerr, marker="o", capsize=4)
            ax.set_xlabel("particle number N")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Finite-size convergence: {ylabel}")
            ax.grid(True, alpha=0.3)
            save_figure(fig, run_root / filename)

        # Direct Level-2.6 particle versus Level-2.8 gap-SERS comparison.
        particle_mean = np.array([
            r["particle_downstream_enhancement_probability_mean"]
            for r in ensemble_rows
        ])
        particle_std = np.array([
            r["particle_downstream_enhancement_probability_std"]
            for r in ensemble_rows
        ])
        gap_mean = np.array([
            r["gap_SERS_downstream_enhancement_probability_mean"]
            for r in ensemble_rows
        ])
        gap_std = np.array([
            r["gap_SERS_downstream_enhancement_probability_std"]
            for r in ensemble_rows
        ])
        fig, ax = plt.subplots(figsize=(8, 5.4))
        ax.errorbar(Ns, particle_mean, yerr=particle_std, marker="o", capsize=4,
                    linewidth=1.8, label="particle dipole, whole-network edges")
        ax.errorbar(Ns, gap_mean, yerr=gap_std, marker="s", capsize=4,
                    linewidth=1.8, label="gap-SERS, adjacent gap pairs")
        ax.set_xlabel("particle number N")
        ax.set_ylabel("downstream enhancement probability")
        ax.set_title("Ensemble downstream enhancement probability vs N")
        ax.set_xticks(Ns)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        save_figure(fig, run_root / "ensemble_enhancement_probability_vs_N")

    print("\nStudy complete.")
    print(f"Results saved to: {run_root}")

    if SHOW_SUMMARY_FIGURES:
        # Re-open the key PNG figures using matplotlib only if present.
        # This keeps the script convenient in local Jupyter/VS Code.
        candidates = [
            run_root / "convergence_L1e.png",
            run_root / "convergence_L1pct.png",
            run_root / "convergence_dGmax.png",
        ]
        candidates = [p for p in candidates if p.exists()]
        if candidates:
            for p in candidates:
                img = plt.imread(p)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(p.name)
            plt.show()

    return all_rows, ensemble_rows


# =============================================================================
# 10. MAIN
# =============================================================================
if __name__ == "__main__":
    run_study()
