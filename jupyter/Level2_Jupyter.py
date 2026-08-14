# %% [markdown]
# # Level 2.6 — validated 2D coupled-dipole network
#
# This percent-format file is both an ordinary Python script and a notebook
# source.  It delegates the numerical run to
# `stochastic_2d.validation_level2_6` so the validated equations, defaults,
# solver, and calculations remain unchanged.

# %%
# Imports
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
if not (PROJECT_ROOT / "stochastic_2d").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stochastic_2d.validation_level2_6 import run as run_validated_level2

# %% [markdown]
# ## Physical constants
# The following values document the constants frozen in the validated run.

# %%
PHYSICAL_CONSTANTS = {
    "wavelength_nm": 633.0,
    "medium_relative_permittivity": 1.0 + 0.0j,
    "incident_amplitude_v_per_m": 1.0 + 0.0j,
}
pd.DataFrame(PHYSICAL_CONSTANTS.items(), columns=["constant", "value"])

# %% [markdown]
# ## Model parameters

# %%
MODEL_PARAMETERS = {
    "n": 200,
    "seed": 20260813,
    "remote_intensity_threshold": 1e-6,
    "remote_distance_nm": 500.0,
    "minimum_steps": 15,
}
OUTPUT_DIR = PROJECT_ROOT / "results" / "level2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(MODEL_PARAMETERS.items(), columns=["parameter", "value"])

# %% [markdown]
# ## Material parameters
# Classical spherical Ag particles use the unchanged illustrative relative
# permittivity.  This value is not an experimentally calibrated dielectric
# dataset, as documented by the validated model.

# %%
MATERIAL_PARAMETERS = {
    "silver_relative_permittivity": -15.0 + 1.0j,
    "particle_radius_nm": 10.0,
    "beam_waist_nm": 55.0,
}
pd.DataFrame(MATERIAL_PARAMETERS.items(), columns=["material_parameter", "value"])

# %% [markdown]
# ## Nanoparticle/network geometry
# The validated routine grows the seeded, connected, non-overlapping aggregate
# and identifies every physical near-contact edge.  Geometry is not recreated
# or approximated in this notebook layer.

# %% [markdown]
# ## CDA interaction calculation
# The routine assembles the full complex, retarded 3D electric-dyadic matrix for
# every particle pair; graph edges do not truncate electromagnetic coupling.

# %% [markdown]
# ## Dipole solution
# The unchanged dense complex system is solved for all induced particle dipoles.

# %%
summary = run_validated_level2(output_dir=OUTPUT_DIR, **MODEL_PARAMETERS)
tables = {
    name: pd.read_csv(OUTPUT_DIR / "data" / f"{name}.csv")
    for name in ("summary", "particles", "edges", "path", "enhancement_events")
}
tables["particles"].head(10)

# %% [markdown]
# ## Gap-field / Egap calculation
# Level 2.6 saves incident, scattered, and total fields at particle centers.
# A gap-centered `Egap` calculation is **not part of validated Level 2.6**; it
# first appears in the validated Level 3 hotspot/full-wave workflows.  No new
# gap-field equation is invented here.

# %%
complex_fields = pd.read_csv(OUTPUT_DIR / "data" / "complex_fields.csv")
complex_fields.head(9)

# %% [markdown]
# ## Hotspot identification
# Level 2.6 identifies near-contact network edges, not Level 3 gap hotspots.
# The edge table below is the complete validated Level 2 near-contact input that
# the subsequent level uses.

# %%
tables["edges"].head(10)

# %% [markdown]
# ## Propagation analysis
# The selected long geodesic path and all naturally increasing downstream edges
# are reported without fitting an attenuation law.

# %%
tables["path"]

# %%
tables["enhancement_events"].head(10)

# %% [markdown]
# ## Ensemble analysis
# The validated Level 2.6 result is one seeded realization, not an ensemble.
# Distribution tables are displayed without adding unsupported ensemble claims.

# %%
distribution_tables = {
    name: pd.read_csv(OUTPUT_DIR / "data" / f"{name}.csv")
    for name in ("radius_values", "gap_values", "coordination_summary")
}
distribution_tables["coordination_summary"]

# %% [markdown]
# ## Effective propagation quantities
# The validated model reports path distance, excitation, downstream-event
# probability, and remote classification.  It does not fit a macroscopic
# propagation or attenuation coefficient.

# %%
effective_propagation = tables["summary"][tables["summary"]["metric"].isin([
    "target_geodesic_nm", "target_geodesic_steps", "target_dipole_normalized",
    "enhancement_probability", "remote_particles",
])]
effective_propagation

# %% [markdown]
# ## Figures
# Figures returned below remain open for inline display and are also saved as
# 300 dpi PNG and vector PDF files in `results/level2`.

# %%
particles, edges = tables["particles"], tables["edges"]
fig, ax = plt.subplots(figsize=(8, 6))
for edge in edges.itertuples():
    xy = particles.loc[[edge.i, edge.j], ["x_nm", "y_nm"]]
    ax.plot(xy.x_nm, xy.y_nm, color="0.82", linewidth=0.45, zorder=1)
values = particles.dipole_normalized.clip(lower=1e-12)
points = ax.scatter(particles.x_nm, particles.y_nm, c=values, s=24,
                    norm=LogNorm(1e-12, 1), cmap="viridis", zorder=2)
fig.colorbar(points, ax=ax, label="normalized dipole intensity")
ax.set(xlabel="x (nm)", ylabel="y (nm)", title="Level 2 coupled-dipole network")
ax.set_aspect("equal"); fig.tight_layout()
fig.savefig(OUTPUT_DIR / "network_dipole_intensity.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "network_dipole_intensity.pdf", bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
path = tables["path"]
ax.plot(path.cumulative_geodesic_nm, path.dipole_normalized.clip(lower=1e-300),
        "o-", markersize=3)
ax.set_yscale("log")
ax.set(xlabel="cumulative geodesic distance (nm)",
       ylabel="normalized dipole intensity", title="Remote transport path")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "remote_transport_path.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "remote_transport_path.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## CSV outputs

# %%
csv_outputs = pd.DataFrame({
    "csv_file": sorted(str(path.relative_to(PROJECT_ROOT)) for path in (OUTPUT_DIR / "data").glob("*.csv"))
})
csv_outputs

# %% [markdown]
# ## Summary

# %%
tables["summary"]
print("Level 2.6 completed successfully; outputs:", OUTPUT_DIR.relative_to(PROJECT_ROOT))
