"""Reproducible generation of connected, non-overlapping 2D aggregates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


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
    positions_nm: NDArray[np.float64]
    radii_nm: NDArray[np.float64]
    edges: NDArray[np.int64]
    edge_gaps_nm: NDArray[np.float64]
    seed: int

    @property
    def n_particles(self) -> int:
        return len(self.radii_nm)

    @property
    def degrees(self) -> NDArray[np.int64]:
        values = np.zeros(self.n_particles, dtype=np.int64)
        if len(self.edges):
            np.add.at(values, self.edges[:, 0], 1)
            np.add.at(values, self.edges[:, 1], 1)
        return values


def _truncated_normal(
    rng: np.random.Generator, mean: float, std: float, low: float, high: float
) -> float:
    if std < 0 or low > high:
        raise ValueError("invalid distribution parameters")
    if std == 0:
        if not low <= mean <= high:
            raise ValueError("constant distribution lies outside its bounds")
        return mean
    for _ in range(10_000):
        value = float(rng.normal(mean, std))
        if low <= value <= high:
            return value
    raise RuntimeError("could not sample the requested truncated distribution")


def _validate_config(config: GeometryConfig) -> None:
    if config.n_particles < 1:
        raise ValueError("n_particles must be positive")
    if config.radius_min_nm <= 0 or config.gap_min_nm < 0:
        raise ValueError("radii must be positive and gaps non-negative")
    if config.connection_gap_nm < config.gap_max_nm:
        raise ValueError("connection_gap_nm must cover every generated attachment gap")


def generate_aggregate(config: GeometryConfig = GeometryConfig()) -> Aggregate:
    """Grow a compact disordered aggregate and infer all physical near contacts.

    Every new particle is attached to an existing particle, guaranteeing a
    connected backbone.  Random parent choice, angular disorder, and alternating
    compact/branch growth create heterogeneous density, while additional near
    contacts close loops. Surface distances are checked against every prior
    particle so overlaps cannot be accepted.
    """
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    radii = np.array([
        _truncated_normal(rng, config.radius_mean_nm, config.radius_std_nm,
                          config.radius_min_nm, config.radius_max_nm)
        for _ in range(config.n_particles)
    ])
    positions = np.zeros((config.n_particles, 2), dtype=float)
    attachment_edges: list[tuple[int, int]] = []
    degrees = np.zeros(config.n_particles, dtype=int)

    for child in range(1, config.n_particles):
        placed = False
        for attempt in range(config.max_attempts_per_particle):
            existing = np.arange(child)
            # Most growth favours low degree (branches); periodic compact growth
            # favours the centroid and increases incidental contacts/loops.
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
            gap = _truncated_normal(rng, config.gap_mean_nm, config.gap_std_nm,
                                    config.gap_min_nm, config.gap_max_nm)
            separation = radii[parent] + radii[child] + gap
            candidate = positions[parent] + separation * np.array([math.cos(angle), math.sin(angle)])
            center_distances = np.linalg.norm(positions[:child] - candidate, axis=1)
            # Maintain the configured classical-PDA minimum surface gap against
            # every particle, including incidental contacts.
            required = radii[:child] + radii[child] + config.gap_min_nm
            if np.all(center_distances >= required - 1e-10):
                positions[child] = candidate
                attachment_edges.append((parent, child))
                degrees[parent] += 1
                degrees[child] += 1
                placed = True
                break
        if not placed:
            raise RuntimeError(f"failed to place particle {child}; relax geometry settings")

    edges: list[tuple[int, int]] = []
    gaps: list[float] = []
    for i in range(config.n_particles):
        offsets = positions[i + 1:] - positions[i]
        surface_gaps = np.linalg.norm(offsets, axis=1) - radii[i] - radii[i + 1:]
        for relative_j in np.flatnonzero(surface_gaps <= config.connection_gap_nm + 1e-9):
            j = i + 1 + int(relative_j)
            edges.append((i, j))
            gaps.append(max(0.0, float(surface_gaps[relative_j])))

    aggregate = Aggregate(positions, radii, np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                          np.asarray(gaps), config.seed)
    attachment_set = {tuple(sorted(edge)) for edge in attachment_edges}
    if not attachment_set.issubset({tuple(edge) for edge in edges}):
        raise RuntimeError("internal error: connected backbone was lost")
    return aggregate


def minimum_surface_gap(aggregate: Aggregate) -> float:
    """Return the smallest surface separation among all particle pairs."""
    minimum = math.inf
    for i in range(aggregate.n_particles - 1):
        distances = np.linalg.norm(aggregate.positions_nm[i + 1:] - aggregate.positions_nm[i], axis=1)
        gaps = distances - aggregate.radii_nm[i] - aggregate.radii_nm[i + 1:]
        minimum = min(minimum, float(gaps.min()))
    return minimum
