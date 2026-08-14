"""Run Level 3A: uncalibrated gap-hotspot network."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from .coupled_dipole import solve_coupled_dipoles
from .excitation import gaussian_incident_field
from .geometry import GeometryConfig,generate_aggregate
from .hotspots import construct_hotspots,fields_at_hotspots,hotspot_distances,classify_hotspots
from .polarizability import sphere_polarizability

def _write(path,header,rows):
    with path.open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(header);w.writerows(rows)
def _save(fig,path):
    fig.tight_layout();fig.savefig(path,format="svg");plt.close(fig)
    path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines())+"\n")
def _field_parts(field):return [x for c in field for x in (c.real,c.imag)]

def run(output_dir="results/level3A",n=200,seed=20260813,gap_min_nm=1.,gap_max_nm=6.,
        incident_threshold=1e-6,remote_distance_nm=500.):
    out=Path(output_dir);data=out/"data";data.mkdir(parents=True,exist_ok=True)
    a=generate_aggregate(GeometryConfig(n_particles=n,seed=seed));h=construct_hotspots(a,gap_min_nm,gap_max_nm)
    pos=np.c_[a.positions_nm,np.zeros(n)]*1e-9; wavelength=633.; eps=-15+1j;k=2*np.pi/(wavelength*1e-9)
    source=int(np.argmin(a.positions_nm[:,0]));center=a.positions_nm[source];waist=55.;e0=1+0j
    pinc=gaussian_incident_field(pos,np.r_[center,0]*1e-9,waist*1e-9,e0,np.array([1.,0.,0.]))
    alpha=sphere_polarizability(a.radii_nm*1e-9,wavelength*1e-9,eps)
    sol=solve_coupled_dipoles(pos,alpha,pinc,k)
    inc,scat,total=fields_at_hotspots(h,pos,sol.dipoles_c_m,k,np.r_[center,0]*1e-9,waist*1e-9)
    m2i=np.sum(abs(inc)**2,1)/abs(e0)**2;m2s=np.sum(abs(scat)**2,1)/abs(e0)**2
    m2=np.sum(abs(total)**2,1)/abs(e0)**2;m4=m2**2
    euclid=np.linalg.norm(h.positions_nm-center,axis=1);labels=classify_hotspots(m2i,euclid,incident_threshold,remote_distance_nm)
    source_nodes=np.flatnonzero(labels=="source");steps,geodesic=hotspot_distances(h,source_nodes)
    source_max=int(source_nodes[np.argmax(m4[source_nodes])]);remote_nodes=np.flatnonzero(labels=="remote")
    if not len(remote_nodes):raise RuntimeError("no remote hotspots")
    remote_max=int(remote_nodes[np.argmax(m4[remote_nodes])])
    finc=np.divide(m2i,m2,out=np.full_like(m2,np.nan),where=m2>0);fscat=np.divide(m2s,m2,out=np.full_like(m2,np.nan),where=m2>0)

    transitions=[]
    for u,neighbors in enumerate(h.adjacency):
        for v in neighbors:
            if u>=v or geodesic[u]==geodesic[v]:continue
            upstream,downstream=(u,v) if geodesic[u]<geodesic[v] else (v,u)
            ratio=m4[downstream]/m4[upstream] if m4[upstream]>0 else np.inf
            transitions.append((upstream,downstream,geodesic[upstream],geodesic[downstream],m4[upstream],m4[downstream],ratio,ratio>1,labels[downstream]))
    enhancements=[x for x in transitions if x[7]];remote_events=[x for x in enhancements if x[8]=="remote"]
    strongest_remote_event=max(remote_events,key=lambda x:x[6]) if remote_events else None

    fields=["x_real","x_imag","y_real","y_imag","z_real","z_imag"]
    header=["hotspot_id","particle_i","particle_j","x_nm","y_nm","gap_nm","radius_i_nm","radius_j_nm","coordination_i","coordination_j","euclidean_distance_nm","geodesic_distance_nm","geodesic_steps",*["Einc_"+x for x in fields],*["Escat_"+x for x in fields],*["Etotal_"+x for x in fields],"M2_inc","M2_scat","M2_total","M4","f_inc","f_scat","remote_classification"]
    rows=[]
    for q,(i,j) in enumerate(h.pairs):rows.append([q,i,j,*h.positions_nm[q],h.gaps_nm[q],a.radii_nm[i],a.radii_nm[j],a.degrees[i],a.degrees[j],euclid[q],geodesic[q],steps[q],*_field_parts(inc[q]),*_field_parts(scat[q]),*_field_parts(total[q]),m2i[q],m2s[q],m2[q],m4[q],finc[q],fscat[q],labels[q]])
    _write(data/"hotspots.csv",header,rows)
    particle_adj=[set() for _ in range(n)]
    for i,j in a.edges:particle_adj[int(i)].add(int(j));particle_adj[int(j)].add(int(i))
    def in_particle_loop(pair):
        start,target=map(int,pair);seen={start};queue=[start]
        for node in queue:
            for nxt in particle_adj[node]:
                if {node,nxt}=={start,target}:continue
                if nxt==target:return True
                if nxt not in seen:seen.add(nxt);queue.append(nxt)
        return False
    event_header=["upstream_hotspot","downstream_hotspot","upstream_particles","downstream_particles","upstream_x_nm","upstream_y_nm","downstream_x_nm","downstream_y_nm","upstream_gap_nm","downstream_gap_nm","upstream_euclidean_nm","downstream_euclidean_nm","upstream_geodesic_nm","downstream_geodesic_nm","upstream_coordination","downstream_coordination","upstream_Etotal_components","downstream_Etotal_components","upstream_M2","downstream_M2","upstream_M4","downstream_M4","ratio","associated_branch","associated_loop","high_coordination","unusually_short_gap","destination_class"]
    detailed=[]
    short_cut=np.percentile(h.gaps_nm,25)
    for u,v,*rest in enhancements:
        upcoord=a.degrees[h.pairs[u]].tolist();downcoord=a.degrees[h.pairs[v]].tolist()
        detailed.append([u,v,h.pairs[u].tolist(),h.pairs[v].tolist(),*h.positions_nm[u],*h.positions_nm[v],h.gaps_nm[u],h.gaps_nm[v],euclid[u],euclid[v],geodesic[u],geodesic[v],upcoord,downcoord,_field_parts(total[u]),_field_parts(total[v]),m2[u],m2[v],m4[u],m4[v],m4[v]/m4[u],max(upcoord+downcoord)>=3,in_particle_loop(h.pairs[u]) or in_particle_loop(h.pairs[v]),max(upcoord+downcoord)>=4,min(h.gaps_nm[u],h.gaps_nm[v])<short_cut,labels[v]])
    _write(data/"downstream_hotspot_enhancements.csv",event_header,detailed)
    if strongest_remote_event:
        u,v=strongest_remote_event[:2]
        _write(data/"strongest_remote_enhancement.csv",["role","hotspot_id","particle_i","particle_j","x_nm","y_nm","gap_nm","euclidean_nm","geodesic_nm","coordination_i","coordination_j","Etotal_x_real","Etotal_x_imag","Etotal_y_real","Etotal_y_imag","Etotal_z_real","Etotal_z_imag","M2","M4","event_ratio"],
               ([role,q,*h.pairs[q],*h.positions_nm[q],h.gaps_nm[q],euclid[q],geodesic[q],a.degrees[h.pairs[q,0]],a.degrees[h.pairs[q,1]],*_field_parts(total[q]),m2[q],m4[q],strongest_remote_event[6]] for role,q in (("upstream",u),("downstream",v))))

    stats=[]
    for metric,distances in (("euclidean",euclid),("geodesic",geodesic)):
        edges=np.linspace(distances.min(),distances.max(),9)
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(distances>=lo)&(distances<hi if hi<edges[-1] else distances<=hi)
            if not mask.any():continue
            for name,val in (("M2",m2),("M4",m4)):
                stats.append([metric,lo,hi,name,mask.sum(),np.mean(val[mask]),np.median(val[mask]),np.percentile(val[mask],95),np.max(val[mask])])
    _write(data/"hotspot_distance_statistics.csv",["distance_type","bin_min_nm","bin_max_nm","metric","count","mean","median","percentile_95","maximum"],stats)

    particle_intensity=np.sum(abs(sol.dipoles_c_m)**2,1);particle_norm=particle_intensity/particle_intensity.max()
    adjacent_particle=np.maximum(particle_norm[h.pairs[:,0]],particle_norm[h.pairs[:,1]])
    comparison=np.c_[np.arange(len(h.pairs)),adjacent_particle,m4/m4.max()]
    _write(data/"particle_hotspot_comparison.csv",["hotspot_id","max_adjacent_particle_intensity_normalized","hotspot_M4_normalized"],comparison)

    def scattermap(name,values,title,normlog=True):
        fig,ax=plt.subplots(figsize=(8,6));ax.set_aspect("equal");kwargs={}
        if normlog:kwargs["norm"]=LogNorm(vmin=max(values.max()*1e-12,np.finfo(float).tiny),vmax=values.max())
        sc=ax.scatter(*h.positions_nm.T,c=np.maximum(values,values.max()*1e-12) if normlog else values,s=18,cmap="magma",**kwargs);fig.colorbar(sc,ax=ax,label=title);ax.set(title=title,xlabel="x (nm)",ylabel="y (nm)");_save(fig,out/name)
    fig,ax=plt.subplots(figsize=(8,6));ax.set_aspect("equal")
    for p,r in zip(a.positions_nm,a.radii_nm):ax.add_patch(plt.Circle(p,r,fc="none",ec="#aaa",lw=.3))
    ax.scatter(*h.positions_nm.T,s=8,c="red");ax.autoscale();ax.set_title("Physical gap hotspot positions");_save(fig,out/"hotspot_geometry.svg")
    scattermap("hotspot_gap_map.svg",h.gaps_nm,"surface gap (nm)",False);scattermap("hotspot_M2_map.svg",m2,"M2")
    scattermap("hotspot_M4_map.svg",m4/m4.max(),"uncalibrated M4 / maximum")
    scattermap("incident_hotspot_field.svg",m2i,"M2 incident");scattermap("scattered_hotspot_field.svg",m2s,"M2 scattered");scattermap("total_hotspot_field.svg",m2,"M2 total")
    fig,ax=plt.subplots(figsize=(8,6));colors={"source":"orange","transition":"gray","remote":"blue"}
    for label in colors:
        mask=labels==label;ax.scatter(*h.positions_nm[mask].T,s=15,c=colors[label],label=label)
    ax.scatter(*h.positions_nm[remote_max],s=100,marker="*",c="red");ax.legend();ax.set_aspect("equal");_save(fig,out/"remote_hotspot_map.svg")
    for x,name,xlabel in ((euclid,"M4_vs_euclidean_distance.svg","Euclidean distance (nm)"),(geodesic,"M4_vs_geodesic_distance.svg","geodesic distance (nm)")):
        fig,ax=plt.subplots();ax.scatter(x,np.log10(np.maximum(m4,1e-300)),s=10);ax.set(xlabel=xlabel,ylabel="log10 uncalibrated M4");_save(fig,out/name)
    for x,name,xlabel in ((euclid,"M2_vs_euclidean_distance.svg","Euclidean distance (nm)"),(geodesic,"M2_vs_geodesic_distance.svg","geodesic distance (nm)")):
        fig,ax=plt.subplots();ax.scatter(x,np.log10(np.maximum(m2,1e-300)),s=10);ax.set(xlabel=xlabel,ylabel="log10 M2");_save(fig,out/name)
    fig,axes=plt.subplots(1,2,figsize=(11,4));
    for ax,kind in zip(axes,("euclidean","geodesic")):
        subset=[r for r in stats if r[0]==kind and r[3]=="M4"];x=[(r[1]+r[2])/2 for r in subset]
        for idx,label in ((5,"mean"),(6,"median"),(7,"95th percentile"),(8,"maximum")):ax.plot(x,[r[idx] for r in subset],label=label)
        ax.set_yscale("log");ax.legend();ax.set_title(kind)
    _save(fig,out/"hotspot_statistics_vs_distance.svg")
    fig,ax=plt.subplots(figsize=(8,6));ax.scatter(*h.positions_nm.T,s=6,c="#bbb")
    for u,v,*_ in enhancements:ax.plot(*h.positions_nm[[u,v]].T,c="red",lw=.5)
    ax.set_aspect("equal");ax.set_title("Natural downstream hotspot enhancements");_save(fig,out/"downstream_hotspot_enhancement.svg")
    i,j=h.pairs[remote_max];fig,ax=plt.subplots();
    local=set([int(i),int(j)]+[int(x) for edge in a.edges if i in edge or j in edge for x in edge])
    for p in local:ax.add_patch(plt.Circle(a.positions_nm[p],a.radii_nm[p],fc="none",ec="black"));ax.text(*a.positions_nm[p],str(p),ha="center")
    ax.scatter(*h.positions_nm[remote_max],marker="*",s=120,c="red");ax.autoscale();ax.set_aspect("equal");ax.set_title(f"Hotspot {remote_max}: gap={h.gaps_nm[remote_max]:.3f} nm, M4={m4[remote_max]:.3g}");_save(fig,out/"strongest_remote_hotspot.svg")

    moderate_strong=int(np.argmax((m4/m4.max())/(adjacent_particle+1e-30)));strong_weak=int(np.argmax(adjacent_particle/(m4/m4.max()+1e-30)))
    ratios=np.array([x[6] for x in enhancements]); summary={"particles":n,"near_contact_gaps":len(a.edges),"hotspots":len(h.pairs),"gap_minimum_threshold_nm":gap_min_nm,"gap_maximum_threshold_nm":gap_max_nm,"gap_mean_nm":h.gaps_nm.mean(),"gap_median_nm":np.median(h.gaps_nm),"gap_min_nm":h.gaps_nm.min(),"gap_max_nm":h.gaps_nm.max(),"source_hotspots":np.sum(labels=="source"),"transition_hotspots":np.sum(labels=="transition"),"remote_hotspots":len(remote_nodes),"remote_percent":100*len(remote_nodes)/len(h.pairs),"strongest_source_hotspot":source_max,"source_particles":h.pairs[source_max].tolist(),"source_coordinates_nm":h.positions_nm[source_max].tolist(),"source_gap_nm":h.gaps_nm[source_max],"source_euclidean_nm":euclid[source_max],"source_geodesic_nm":geodesic[source_max],"source_M2":m2[source_max],"source_M4":m4[source_max],"strongest_remote_hotspot":remote_max,"remote_particles":h.pairs[remote_max].tolist(),"remote_coordinates_nm":h.positions_nm[remote_max].tolist(),"remote_coordination":a.degrees[h.pairs[remote_max]].tolist(),"remote_gap_nm":h.gaps_nm[remote_max],"remote_euclidean_nm":euclid[remote_max],"remote_geodesic_nm":geodesic[remote_max],"remote_geodesic_steps":steps[remote_max],"remote_M2_inc":m2i[remote_max],"remote_M2_scat":m2s[remote_max],"remote_M2_total":m2[remote_max],"remote_M4":m4[remote_max],"remote_incident_to_scattered_ratio":m2i[remote_max]/m2s[remote_max],"remote_source_M2_ratio":m2[remote_max]/m2[source_max],"remote_source_M4_ratio":m4[remote_max]/m4[source_max],"downstream_transitions":len(transitions),"enhancement_events":len(enhancements),"enhancement_probability":len(enhancements)/len(transitions),"maximum_enhancement_ratio":ratios.max(),"median_enhancement_ratio":np.median(ratios),"percentile95_enhancement_ratio":np.percentile(ratios,95),"strongest_remote_event":str(strongest_remote_event),"moderate_particle_strong_hotspot_example":moderate_strong,"strong_particle_weak_hotspot_example":strong_weak,"strongest_hotspot_adjacent_to_strongest_particle":bool(np.argmax(particle_intensity) in h.pairs[np.argmax(m4)]),"all_particle_contributions_per_hotspot":n,"matrix_finite":np.isfinite(sol.matrix).all(),"hotspot_fields_finite":np.isfinite(total).all()}
    _write(data/"summary.csv",["metric","value"],summary.items());return summary

if __name__=="__main__":
    for k,v in run().items():print(f"{k}: {v}")
