"""Uncalibrated, gap-centered hotspot construction and field evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np
from numpy.typing import NDArray

from .excitation import gaussian_incident_field
from .green_tensor import electric_dyadic
from .geometry import Aggregate


@dataclass(frozen=True)
class HotspotData:
    pairs: NDArray[np.int64]
    positions_nm: NDArray[np.float64]
    gaps_nm: NDArray[np.float64]
    adjacency: tuple[tuple[int, ...], ...]


def construct_hotspots(aggregate: Aggregate, gap_min_nm: float = 1.0,
                       gap_max_nm: float = 6.0) -> HotspotData:
    """Create hotspot nodes only for geometrically valid near-contact gaps."""
    pairs=[]; positions=[]; gaps=[]
    for i,j in aggregate.edges:
        i,j=int(i),int(j); delta=aggregate.positions_nm[j]-aggregate.positions_nm[i]
        distance=float(np.linalg.norm(delta)); gap=distance-aggregate.radii_nm[i]-aggregate.radii_nm[j]
        if gap_min_nm-1e-9 <= gap <= gap_max_nm+1e-9:
            unit=delta/distance
            surface_i=aggregate.positions_nm[i]+aggregate.radii_nm[i]*unit
            surface_j=aggregate.positions_nm[j]-aggregate.radii_nm[j]*unit
            pairs.append((i,j)); positions.append((surface_i+surface_j)/2); gaps.append(gap)
    if not pairs: raise ValueError("no particle pairs satisfy the hotspot gap bounds")
    adjacency=[set() for _ in pairs]; memberships={}
    for h,pair in enumerate(pairs):
        for particle in pair:
            for other in memberships.get(particle,[]): adjacency[h].add(other);adjacency[other].add(h)
            memberships.setdefault(particle,[]).append(h)
    return HotspotData(np.asarray(pairs,dtype=np.int64),np.asarray(positions),np.asarray(gaps),
                       tuple(tuple(sorted(x)) for x in adjacency))


def fields_at_hotspots(hotspots: HotspotData, particle_positions_m, dipoles,
                       wave_number, source_center_m, waist_m, amplitude=1+0j,
                       polarization=np.array([1.,0.,0.]), medium_permittivity=1+0j):
    """Evaluate complex incident/scattered/total fields using every dipole."""
    locations=np.c_[hotspots.positions_nm,np.zeros(len(hotspots.pairs))]*1e-9
    incident=gaussian_incident_field(locations,source_center_m,waist_m,amplitude,polarization)
    scattered=np.zeros_like(incident,dtype=complex)
    for h,location in enumerate(locations):
        for center,dipole in zip(particle_positions_m,dipoles):
            scattered[h]+=electric_dyadic(location-center,wave_number,medium_permittivity)@dipole
    return incident,scattered,incident+scattered


def hotspot_distances(hotspots: HotspotData, source_hotspots: NDArray[np.int64]):
    """Return minimum graph steps and physically weighted hotspot geodesics."""
    count=len(hotspots.pairs); steps=np.full(count,-1,int); lengths=np.full(count,np.inf)
    queue=[]
    for source in source_hotspots:
        steps[source]=0;lengths[source]=0.;heapq.heappush(queue,(0.,int(source)))
    while queue:
        distance,node=heapq.heappop(queue)
        if distance!=lengths[node]:continue
        for neighbor in hotspots.adjacency[node]:
            candidate=distance+np.linalg.norm(hotspots.positions_nm[node]-hotspots.positions_nm[neighbor])
            if candidate<lengths[neighbor]:
                lengths[neighbor]=candidate;steps[neighbor]=steps[node]+1
                heapq.heappush(queue,(candidate,neighbor))
    return steps,lengths


def classify_hotspots(incident_normalized, euclidean_nm, incident_threshold=1e-6,
                      remote_distance_nm=500.):
    remote=(incident_normalized<incident_threshold)&(euclidean_nm>remote_distance_nm)
    source=incident_normalized>=incident_threshold
    return np.where(remote,"remote",np.where(source,"source","transition"))
