"""Level 2.6 physical-consistency and remote-propagation validation."""
from __future__ import annotations

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


def _graph(edges, n, source):
    adj=[[] for _ in range(n)]
    gap={}
    for k,(i,j) in enumerate(edges):
        i,j=int(i),int(j); adj[i].append(j); adj[j].append(i); gap[tuple(sorted((i,j)))]=k
    dist=np.full(n,-1,int); parent=np.full(n,-1,int); dist[source]=0; q=[source]
    for u in q:
        for v in sorted(adj[u]):
            if dist[v]<0: dist[v]=dist[u]+1; parent[v]=u; q.append(v)
    return adj,gap,dist,parent


def _path(parent, target):
    p=[target]
    while parent[p[-1]]>=0: p.append(int(parent[p[-1]]))
    return np.array(p[::-1])


def _savefig(fig,path):
    fig.tight_layout(); fig.savefig(path,format="svg"); plt.close(fig)
    path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines())+"\n")


def run(output_dir="results/level2_6", n=200, seed=20260813,
        remote_intensity_threshold=1e-6, remote_distance_nm=500., minimum_steps=15):
    out=Path(output_dir); data=out/"data"; data.mkdir(parents=True,exist_ok=True)
    cfg=GeometryConfig(n_particles=n,seed=seed, radius_mean_nm=10,radius_std_nm=0,
        radius_min_nm=10,radius_max_nm=10,gap_mean_nm=3,gap_std_nm=1,
        gap_min_nm=1,gap_max_nm=6,connection_gap_nm=6)
    a=generate_aggregate(cfg); pos=np.c_[a.positions_nm,np.zeros(n)]*1e-9
    wavelength_nm=633.; eps_ag=-15+1j; eps_m=1+0j; waist_nm=55.; amplitude=1+0j
    source=int(np.argmin(a.positions_nm[:,0])); center=a.positions_nm[source]
    inc=gaussian_incident_field(pos,np.r_[center,0]*1e-9,waist_nm*1e-9,amplitude,
                                np.array([1.,0.,0.]))
    k=2*np.pi/(wavelength_nm*1e-9)
    alpha=sphere_polarizability(a.radii_nm*1e-9,wavelength_nm*1e-9,eps_ag,eps_m)
    sol=solve_coupled_dipoles(pos,alpha,inc,k,eps_m)
    scat=scattered_field(pos,sol.dipoles_c_m,k,eps_m); total=inc+scat
    ii=np.sum(abs(inc)**2,1); si=np.sum(abs(scat)**2,1); ti=np.sum(abs(total)**2,1)
    pi=np.sum(abs(sol.dipoles_c_m)**2,1)
    iin=ii/ii.max(); sin=si/si.max(); tin=ti/ti.max(); pin=pi/pi.max()
    euclid=np.linalg.norm(a.positions_nm-center,axis=1)
    adj,edge_index,gdist,parent=_graph(a.edges,n,source)
    remote=(iin<remote_intensity_threshold)&(euclid>remote_distance_nm)
    eligible=np.flatnonzero(remote&(gdist>=minimum_steps))
    if not len(eligible): raise RuntimeError("no remote particle has the required geodesic length")
    strongest=int(np.flatnonzero(remote)[np.argmax(pi[remote])])
    target=strongest if gdist[strongest]>=minimum_steps else int(eligible[np.argmax(pi[eligible])])
    path=_path(parent,target)
    edge_lengths=np.linalg.norm(np.diff(a.positions_nm[path],axis=0),axis=1)
    cumulative=np.r_[0,np.cumsum(edge_lengths)]

    def edge_in_loop(u,v):
        seen={u}; q=[u]
        for node in q:
            for nxt in adj[node]:
                if {node,nxt}=={u,v}: continue
                if nxt==v:return True
                if nxt not in seen:seen.add(nxt);q.append(nxt)
        return False
    events=[]; valid=[]
    local_gaps=[[] for _ in range(n)]
    for (i,j),g in zip(a.edges,a.edge_gaps_nm): local_gaps[int(i)].append(g);local_gaps[int(j)].append(g)
    for i,j in a.edges:
        i,j=int(i),int(j)
        if gdist[i]==gdist[j]: continue
        u,v=(i,j) if gdist[i]<gdist[j] else (j,i); ratio=pi[v]/pi[u]
        edge_gap=a.edge_gaps_nm[edge_index[tuple(sorted((u,v)))]]
        row=[u,v,gdist[u],gdist[v],a.degrees[u],a.degrees[v],np.mean(local_gaps[u]),
             np.mean(local_gaps[v]),edge_gap,edge_in_loop(u,v),euclid[v],ratio,ratio>1]
        valid.append(row)
        if ratio>1: events.append(row)
    strongest_event=max(events,key=lambda x:x[9])

    def write(name,header,rows):
        with (data/name).open("w",newline="") as f:
            w=csv.writer(f,lineterminator="\n");w.writerow(header);w.writerows(rows)
    write("particles.csv",["id","x_nm","y_nm","radius_nm","degree","euclidean_nm","graph_distance",
          "incident_intensity","scattered_intensity","total_intensity","dipole_intensity",
          "incident_normalized","scattered_normalized","total_normalized","dipole_normalized","remote"],
          ([i,*a.positions_nm[i],a.radii_nm[i],a.degrees[i],euclid[i],gdist[i],ii[i],si[i],ti[i],pi[i],iin[i],sin[i],tin[i],pin[i],remote[i]] for i in range(n)))
    write("edges.csv",["i","j","surface_gap_nm"],([*e,g] for e,g in zip(a.edges,a.edge_gaps_nm)))
    write("path.csv",["step","id","x_nm","y_nm","euclidean_nm","cumulative_geodesic_nm","dipole_intensity","dipole_normalized","incident_intensity","scattered_intensity","total_intensity"],
          ([s,int(i),*a.positions_nm[i],euclid[i],cumulative[s],pi[i],pin[i],ii[i],si[i],ti[i]] for s,i in enumerate(path)))
    eh=["upstream","downstream","upstream_graph_distance","downstream_graph_distance","upstream_degree","downstream_degree","upstream_mean_gap_nm","downstream_mean_gap_nm","edge_gap_nm","edge_in_loop","downstream_euclidean_nm","enhancement_ratio","enhancement"]
    write("downstream_edges.csv",eh,valid);write("enhancement_events.csv",eh,events)
    write("radius_values.csv",["particle_id","radius_nm"],enumerate(a.radii_nm))
    write("gap_values.csv",["edge_id","surface_gap_nm"],enumerate(a.edge_gaps_nm))
    unique,counts=np.unique(a.degrees,return_counts=True)
    coord=[]
    for z,c in zip(unique,counts):
        subset=[r for r in valid if r[4]==z]; coord.append([z,c,len(subset),sum(r[-1] for r in subset),sum(r[-1] for r in subset)/len(subset) if subset else 0])
    write("coordination_summary.csv",["upstream_degree","particle_count","downstream_edges","enhancements","probability"],coord)
    write("complex_fields.csv",["particle_id","field","x_real","x_imag","y_real","y_imag","z_real","z_imag"],
          ([i,name,*[v for component in field[i] for v in (component.real,component.imag)]]
           for i in range(n) for name,field in (("incident",inc),("scattered",scat),("total",total))))

    seg=np.array([[a.positions_nm[i],a.positions_nm[j]] for i,j in a.edges])
    def spatial(name,values,title,clip=1e-12,marks=None):
        fig,ax=plt.subplots(figsize=(8,6));ax.set_aspect("equal");v=np.maximum(values,clip)
        sc=ax.scatter(*a.positions_nm.T,c=v,s=22,cmap="viridis",norm=LogNorm(vmin=clip,vmax=1));fig.colorbar(sc,ax=ax,label=title)
        if marks is not None: ax.scatter(*a.positions_nm[marks].T,facecolors="none",edgecolors="red",s=55)
        ax.set(title=title,xlabel="x (nm)",ylabel="y (nm)");_savefig(fig,out/name)
    fig,ax=plt.subplots(figsize=(8,6));ax.set_aspect("equal")
    for p,r in zip(a.positions_nm,a.radii_nm):ax.add_patch(plt.Circle(p,r,fc="#bbb",ec="#333",lw=.3))
    ax.autoscale();ax.set(title="Level 2.6 aggregate: R=10 nm",xlabel="x (nm)",ylabel="y (nm)");_savefig(fig,out/"aggregate_R10nm.svg")
    for vals,label,name in [(a.radii_nm,"R (nm)","radius_distribution.svg"),(a.edge_gaps_nm,"g (nm)","gap_distribution.svg")]:
        fig,ax=plt.subplots();ax.hist(vals,bins=12,density=True);ax.set(xlabel=label,ylabel="probability density");_savefig(fig,out/name)
    fig,ax=plt.subplots();ax.bar(unique,counts/counts.sum());ax.set(xlabel="z",ylabel="P(z)");_savefig(fig,out/"coordination_distribution.svg")
    spatial("incident_field_map.svg",iin,"normalized incident intensity",1e-12)
    spatial("scattered_field_map.svg",sin,"normalized scattered intensity")
    spatial("total_field_map.svg",tin,"normalized total intensity")
    spatial("normalized_dipole_map.svg",pin,"normalized dipole intensity")
    spatial("remote_region.svg",np.maximum(iin,1e-12),"remote-region classification",1e-12,np.flatnonzero(remote))
    fig,ax=plt.subplots(figsize=(8,6));ax.add_collection(LineCollection(seg,colors="#ddd",lw=.4));ax.plot(*a.positions_nm[path].T,"o-",ms=3);ax.autoscale();ax.set_aspect("equal");_savefig(fig,out/"long_geodesic_path.svg")
    for x,xlabel,name in [(np.arange(len(path)),"geodesic step","path_excitation_vs_step.svg"),(cumulative,"cumulative geodesic distance (nm)","path_excitation_vs_geodesic_distance.svg")]:
        fig,ax=plt.subplots();ax.plot(x,np.log10(np.maximum(pin[path],1e-300)),"o-");ax.set(xlabel=xlabel,ylabel="log10 normalized dipole intensity");_savefig(fig,out/name)
    fig,ax=plt.subplots(figsize=(8,6));ax.add_collection(LineCollection(seg,colors="#ddd",lw=.4));
    for r in events: ax.plot(*a.positions_nm[[r[0],r[1]]].T,color="red",lw=1)
    ax.autoscale();ax.set_aspect("equal");ax.set_title("Natural downstream enhancements");_savefig(fig,out/"downstream_enhancement_map.svg")
    fig,ax=plt.subplots();ax.bar([r[0] for r in coord],[r[-1] for r in coord]);ax.set(xlabel="upstream coordination",ylabel="enhancement probability");_savefig(fig,out/"enhancement_vs_coordination.svg")

    summary={"N":n,"particle_radius_nm":10.,"gap_mean_nm":a.edge_gaps_nm.mean(),"gap_median_nm":np.median(a.edge_gaps_nm),"gap_std_nm":a.edge_gaps_nm.std(),"gap_min_nm":a.edge_gaps_nm.min(),"gap_max_nm":a.edge_gaps_nm.max(),"mean_coordination":a.degrees.mean(),"connected_components":1,"minimum_all_pair_surface_gap_nm":minimum_surface_gap(a),"wavelength_nm":wavelength_nm,"polarization":"[1,0,0]","beam_waist_nm":waist_nm,"source_particle":source,"source_position_nm":center.tolist(),"incident_amplitude_v_per_m":str(amplitude),"surrounding_refractive_index":1.,"silver_relative_permittivity":str(eps_ag),"silver_source":"UNSOURCED illustrative assumption","remote_incident_threshold":remote_intensity_threshold,"remote_distance_threshold_nm":remote_distance_nm,"all_pair_interaction_blocks":n*(n-1),"matrix_finite":np.isfinite(sol.matrix).all(),"solution_finite":np.isfinite(sol.dipoles_c_m).all(),"remote_particles":remote.sum(),"strongest_remote_particle":strongest,"selected_target_particle":target,"target_x_nm":a.positions_nm[target,0],"target_y_nm":a.positions_nm[target,1],"target_euclidean_nm":euclid[target],"target_geodesic_nm":cumulative[-1],"target_geodesic_steps":len(path)-1,"target_incident_normalized":iin[target],"target_scattered_normalized":sin[target],"target_total_normalized":tin[target],"target_dipole_normalized":pin[target],"downstream_edges":len(valid),"enhancement_events":len(events),"enhancement_probability":len(events)/len(valid),"maximum_enhancement_ratio":strongest_event[11],"strongest_enhancement_downstream_degree":strongest_event[5]}
    write("summary.csv",["metric","value"],summary.items())
    return summary


if __name__=="__main__":
    for k,v in run().items(): print(f"{k}: {v}")
