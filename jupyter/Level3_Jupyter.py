# %% [markdown]
# # Level 3A.5 — validated remote-hotspot transport
#
# This script runs the existing validated Level 3A.5 implementation unchanged.
# It first loads and verifies the required Level 2 realization, ensuring the two
# deterministic validated stages cannot silently be mixed.

# %%
# Imports
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
if not (PROJECT_ROOT / "stochastic_2d").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stochastic_2d.geometry import GeometryConfig, generate_aggregate
from stochastic_2d.validation_level3a_5 import run as run_validated_level3

# %% [markdown]
# ## Physical parameters
# These are the exact Level 3A.5 defaults.  `M4` remains an uncalibrated
# electromagnetic `|E|^4` proxy rather than absolute Raman intensity.

# %%
PHYSICAL_PARAMETERS = {
    "n": 200,
    "seed": 20260813,
    "wavelength_nm": 633.0,
    "silver_relative_permittivity": -15.0 + 1.0j,
    "beam_waist_nm": 55.0,
    "dominance_threshold": 100.0,
}
LEVEL2_DIR = PROJECT_ROOT / "results" / "level2"
OUTPUT_DIR = PROJECT_ROOT / "results" / "level3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(PHYSICAL_PARAMETERS.items(), columns=["parameter", "value"])

# %% [markdown]
# ## Loading Level 2 outputs
# Level 3 automatically reads the Level 2 geometry and summary CSVs.  The
# validated Level 3A.5 source deterministically reconstructs the same seeded
# realization internally, so this compatibility gate checks every coordinate,
# radius, edge, and gap before that unchanged source is invoked.

# %%
required_level2 = [
    LEVEL2_DIR / "data" / "particles.csv",
    LEVEL2_DIR / "data" / "edges.csv",
    LEVEL2_DIR / "data" / "complex_fields.csv",
    LEVEL2_DIR / "data" / "summary.csv",
]
missing = [path for path in required_level2 if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Run jupyter/Level2_Jupyter.py first; missing: {missing}")

level2_particles = pd.read_csv(required_level2[0])
level2_edges = pd.read_csv(required_level2[1])
level2_fields = pd.read_csv(required_level2[2])
level2_summary = pd.read_csv(required_level2[3])
level2_particles.head(10)

# %%
expected = generate_aggregate(GeometryConfig(
    n_particles=PHYSICAL_PARAMETERS["n"], seed=PHYSICAL_PARAMETERS["seed"]
))
if not np.allclose(level2_particles[["x_nm", "y_nm"]], expected.positions_nm, rtol=0, atol=1e-12):
    raise ValueError("Level 2 coordinates do not match the validated Level 3A.5 realization")
if not np.allclose(level2_particles["radius_nm"], expected.radii_nm, rtol=0, atol=1e-12):
    raise ValueError("Level 2 radii do not match the validated Level 3A.5 realization")
if not np.array_equal(level2_edges[["i", "j"]].to_numpy(), expected.edges):
    raise ValueError("Level 2 contact graph does not match the validated Level 3A.5 realization")
if not np.allclose(level2_edges["surface_gap_nm"], expected.edge_gaps_nm, rtol=0, atol=1e-12):
    raise ValueError("Level 2 gaps do not match the validated Level 3A.5 realization")
print("Validated Level 2 compatibility gate passed.")

# %% [markdown]
# ## Coarse-graining
# Each validated 1–6 nm physical gap becomes a hotspot at the midpoint between
# its closest particle surfaces.  Shared-particle hotspot adjacency supplies
# topology only; every solved particle dipole contributes to every hotspot.

# %% [markdown]
# ## Macroscopic propagation model
# Level 3A.5 uses physically weighted hotspot-network geodesics and downstream
# transition statistics.  It explicitly does **not** introduce a continuum or
# centimetre-scale attenuation model, so none is fabricated in this local view.

# %%
level3_summary_dict = run_validated_level3(
    output_dir=OUTPUT_DIR,
    n=PHYSICAL_PARAMETERS["n"],
    seed=PHYSICAL_PARAMETERS["seed"],
    dominance_threshold=PHYSICAL_PARAMETERS["dominance_threshold"],
)
DATA_DIR = OUTPUT_DIR / "data"
hotspot_distances = pd.read_csv(DATA_DIR / "corrected_hotspot_distances.csv")
transitions = pd.read_csv(DATA_DIR / "downstream_enhancement_events.csv")
hotspot_distances.head(10)

# %% [markdown]
# ## Remote-SERS related quantities
# `M2=|E_total|^2/|E0|^2` and `M4=M2^2`.  Remote hotspots additionally require
# negligible direct illumination and the validated geometric distance rule;
# plasmon-dominated reporting uses the unchanged scattered/incident threshold.

# %%
remote_hotspots = hotspot_distances.query("classification == 'remote'").copy()
remote_hotspots.nlargest(10, "M4")

# %% [markdown]
# ## Statistical analysis
# Threshold, corrected-distance, and coordination statistics are exactly the
# CSV products written by the validated Level 3A.5 calculation.

# %%
threshold_statistics = pd.read_csv(DATA_DIR / "enhancement_threshold_summary.csv")
distance_statistics = pd.read_csv(DATA_DIR / "corrected_geodesic_statistics.csv")
coordination_statistics = pd.read_csv(DATA_DIR / "enhancement_probability_vs_coordination.csv")
threshold_statistics

# %%
distance_statistics.head(10)

# %% [markdown]
# ## Figures
# The validated routine writes its complete SVG figure set.  The two central
# plots are reproduced from its output tables for inline Jupyter display and
# saved as publication-quality PNG and PDF without changing any calculation.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for label, group in hotspot_distances.groupby("classification"):
    ax.scatter(group.corrected_geodesic_nm, group.M4.clip(lower=1e-300), s=12, label=label)
ax.set_yscale("log")
ax.set(xlabel="corrected weighted geodesic distance (nm)", ylabel="uncalibrated M4",
       title="Validated Level 3A.5 hotspot propagation")
ax.legend(); fig.tight_layout()
fig.savefig(OUTPUT_DIR / "M4_vs_corrected_geodesic.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "M4_vs_corrected_geodesic.pdf", bbox_inches="tight")
plt.show()

# %%
enhanced = transitions.loc[transitions.ratio > 1, "ratio"]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(np.log10(enhanced), bins=25)
ax.set(xlabel="log10 downstream M4 ratio", ylabel="count",
       title="Natural downstream hotspot enhancements")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "downstream_enhancement_distribution.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "downstream_enhancement_distribution.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## CSV outputs

# %%
csv_outputs = pd.DataFrame({
    "csv_file": sorted(str(path.relative_to(PROJECT_ROOT)) for path in DATA_DIR.glob("*.csv"))
})
csv_outputs

# %% [markdown]
# ## Summary

# %%
summary = pd.read_csv(DATA_DIR / "summary.csv")
summary
print("Level 3A.5 completed successfully; outputs:", OUTPUT_DIR.relative_to(PROJECT_ROOT))
