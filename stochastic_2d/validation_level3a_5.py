"""Level 3A.5 corrected-distance and enhancement-threshold validation."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coupled_dipole import solve_coupled_dipoles
from .excitation import gaussian_incident_field
from .geometry import GeometryConfig,generate_aggregate
from .hotspots import construct_hotspots,fields_at_hotspots,hotspot_distances,classify_hotspots
from .polarizability import sphere_polarizability

THRESHOLDS=(1.,2.,10.,100.)

def enhancement_classes(ratios):
    r=np.asarray(ratios);return {"any":r>1,"2x":r>=2,"10x":r>=10,"100x":r>=100}
def plasmon_remote(remote,m2_inc,m2_scat,threshold=100.):
    dominance=np.divide(m2_scat,m2_inc,out=np.full_like(m2_scat,np.inf),where=m2_inc>0)
    return np.asarray(remote)&(dominance>=threshold),dominance
def _write(path,header,rows):
    with path.open("w",newline="") as f:w=csv.writer(f,lineterminator="\n");w.writerow(header);w.writerows(rows)
def _save(fig,path):
    fig.tight_layout();fig.savefig(path,format="svg");plt.close(fig)
    path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines())+"\n")

def run(output_dir="results/level3A_5",n=200,seed=20260813,dominance_threshold=100.):
    out=Path(output_dir);data=out/"data";data.mkdir(parents=True,exist_ok=True)
    a=generate_aggregate(GeometryConfig(n_particles=n,seed=seed));h=construct_hotspots(a)
    pos=np.c_[a.positions_nm,np.zeros(n)]*1e-9;k=2*np.pi/633e-9;source=int(np.argmin(a.positions_nm[:,0]));center=a.positions_nm[source]
    pinc=gaussian_incident_field(pos,np.r_[center,0]*1e-9,55e-9,1+0j,np.array([1.,0.,0.]))
    alpha=sphere_polarizability(a.radii_nm*1e-9,633e-9,-15+1j);sol=solve_coupled_dipoles(pos,alpha,pinc,k)
    inc,scat,total=fields_at_hotspots(h,pos,sol.dipoles_c_m,k,np.r_[center,0]*1e-9,55e-9)
    m2i=np.sum(abs(inc)**2,1);m2s=np.sum(abs(scat)**2,1);m2=np.sum(abs(total)**2,1);m4=m2**2
    euclid=np.linalg.norm(h.positions_nm-center,axis=1);labels=classify_hotspots(m2i,euclid)
    source_nodes=np.flatnonzero(labels=="source");launch=int(source_nodes[np.argmax(m4[source_nodes])])
    steps,distance=hotspot_distances(h,np.array([launch]));remote=labels=="remote"
    plasma,dominance=plasmon_remote(remote,m2i,m2s,dominance_threshold)
    geometry_remote=int(np.flatnonzero(remote)[np.argmax(m4[remote])]);plasmon_remote_id=int(np.flatnonzero(plasma)[np.argmax(m4[plasma])])

    transitions=[]
    for u,neighbors in enumerate(h.adjacency):
        for v in neighbors:
            if u>=v or distance[u]==distance[v]:continue
            i,j=(u,v) if distance[u]<distance[v] else (v,u);ratio=m4[j]/m4[i];delta=np.log10(ratio)
            ci=a.degrees[h.pairs[i]];cj=a.degrees[h.pairs[j]]
            transitions.append([i,j,distance[i],distance[j],steps[i],steps[j],m4[i],m4[j],ratio,delta,h.pairs[i].tolist(),h.pairs[j].tolist(),ci.tolist(),cj.tolist(),h.gaps_nm[i],h.gaps_nm[j],max(*ci,*cj)>=3,labels[j]])
    ratios=np.array([r[8] for r in transitions]);classes=enhancement_classes(ratios)
    header=["upstream_hotspot","downstream_hotspot","upstream_geodesic_nm","downstream_geodesic_nm","upstream_steps","downstream_steps","upstream_M4","downstream_M4","ratio","delta_log10_M4","upstream_particles","downstream_particles","upstream_coordination","downstream_coordination","upstream_gap_nm","downstream_gap_nm","near_branch","downstream_class"]
    _write(data/"downstream_enhancement_events.csv",header,transitions)
    _write(data/"corrected_hotspot_distances.csv",["hotspot_id","steps","corrected_geodesic_nm","classification","M2","M4"],([i,steps[i],distance[i],labels[i],m2[i],m4[i]] for i in range(len(h.pairs))))
    summaries=[]
    for name,mask in classes.items():
        selected=ratios[mask];rvalid=np.array([r[-1]=="remote" for r in transitions]);rsel=ratios[mask&rvalid]
        summaries.append([name,len(transitions),len(selected),len(selected)/len(transitions),np.median(selected),np.percentile(selected,95),selected.max(),rvalid.sum(),len(rsel),len(rsel)/rvalid.sum(),np.median(rsel) if len(rsel) else np.nan,np.percentile(rsel,95) if len(rsel) else np.nan,rsel.max() if len(rsel) else np.nan])
    _write(data/"enhancement_threshold_summary.csv",["class","all_transitions","events","probability","median_ratio","p95_ratio","max_ratio","remote_transitions","remote_events","remote_probability","remote_median","remote_p95","remote_max"],summaries)
    coords=sorted(set(a.degrees));coordrows=[]
    for z in coords:
        subset=np.array([z in r[12] or z in r[13] for r in transitions]);
        for threshold in (2,10,100):coordrows.append([z,threshold,subset.sum(),np.sum(subset&(ratios>=threshold)),np.mean(ratios[subset]>=threshold) if subset.any() else np.nan])
    _write(data/"enhancement_probability_vs_coordination.csv",["coordination","threshold","transitions","events","probability"],coordrows)

    for values,name,label in ((m2,"M2_vs_corrected_geodesic_distance.svg","M2"),(m4,"M4_vs_corrected_geodesic_distance.svg","M4")):
        fig,ax=plt.subplots();ax.scatter(distance,np.log10(np.maximum(values,1e-300)),s=10);ax.set(xlabel="corrected weighted geodesic distance (nm)",ylabel=f"log10 {label}");_save(fig,out/name)
    def eventmap(name,mask,color,width):
        fig,ax=plt.subplots(figsize=(8,6));ax.scatter(*h.positions_nm.T,s=5,c="#bbb")
        for row in np.asarray(transitions,dtype=object)[mask]:ax.plot(*h.positions_nm[[row[0],row[1]]].T,c=color,lw=width,alpha=.7)
        ax.set_aspect("equal");ax.set_title("Downstream hotspot enhancement relations");_save(fig,out/name)
    eventmap("downstream_enhancement_any.svg",classes["any"],"#5ab4ac",.5);eventmap("downstream_enhancement_2x.svg",classes["2x"],"#fdae61",.8);eventmap("downstream_enhancement_10x.svg",classes["10x"],"#d7191c",1.2);eventmap("downstream_enhancement_100x.svg",classes["100x"],"#7b3294",1.8)
    fig,ax=plt.subplots(figsize=(8,6));ax.scatter(*h.positions_nm.T,s=5,c="#ccc")
    for lo,hi,color,label in ((2,10,"#fdae61","2–10x"),(10,100,"#d7191c","10–100x"),(100,np.inf,"#7b3294","≥100x")):
        for r in transitions:
            if lo<=r[8]<hi:ax.plot(*h.positions_nm[[r[0],r[1]]].T,c=color,lw=1,label=label if label not in ax.get_legend_handles_labels()[1] else None)
    ax.legend();ax.set_aspect("equal");ax.set_title("Non-monotonic downstream hotspot events");_save(fig,out/"downstream_enhancement_classes.svg")
    enhanced=ratios[ratios>1];fig,ax=plt.subplots();ax.hist(np.log10(enhanced),bins=25);ax.set(xlabel="log10 R",ylabel="count");_save(fig,out/"downstream_enhancement_ratio_distribution.svg")
    fig,ax=plt.subplots();x=np.sort(enhanced);ax.step(np.log10(x),np.arange(1,len(x)+1)/len(x));ax.set(xlabel="log10 R",ylabel="CDF");_save(fig,out/"downstream_enhancement_ratio_cdf.svg")
    fig,ax=plt.subplots();
    for t in (2,10,100):rows=[r for r in coordrows if r[1]==t];ax.plot([r[0] for r in rows],[r[-1] for r in rows],"o-",label=f"R≥{t}")
    ax.legend();ax.set(xlabel="local coordination",ylabel="associated enhancement probability");_save(fig,out/"enhancement_probability_vs_coordination.svg")
    edges=np.linspace(0,distance.max(),9);statrows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(distance>=lo)&(distance<hi if hi<edges[-1] else distance<=hi)
        for metric,values in (("M2",m2),("M4",m4)):statrows.append([lo,hi,metric,mask.sum(),np.mean(values[mask]),np.median(values[mask]),np.percentile(values[mask],95),np.max(values[mask])])
    _write(data/"corrected_geodesic_statistics.csv",["bin_min_nm","bin_max_nm","metric","count","mean","median","p95","maximum"],statrows)
    fig,ax=plt.subplots();rows=[r for r in statrows if r[2]=="M4"];x=[(r[0]+r[1])/2 for r in rows]
    for idx,label in ((4,"mean"),(5,"median"),(6,"95th"),(7,"maximum")):ax.plot(x,[r[idx] for r in rows],label=label)
    ax.set_yscale("log");ax.legend();ax.set(xlabel="corrected geodesic distance (nm)",ylabel="M4");_save(fig,out/"M4_statistics_corrected_geodesic.svg")
    q=plasmon_remote_id;i,j=h.pairs[q];fig,ax=plt.subplots()
    local=set([int(i),int(j)]+[int(x) for e in a.edges if i in e or j in e for x in e])
    for p in local:ax.add_patch(plt.Circle(a.positions_nm[p],a.radii_nm[p],fill=False));ax.text(*a.positions_nm[p],str(p),ha="center")
    ax.scatter(*h.positions_nm[q],marker="*",s=120,c="red");ax.autoscale();ax.set_aspect("equal");ax.set_title(f"Plasmon-dominated remote hotspot {q}, gap {h.gaps_nm[q]:.3f} nm");_save(fig,out/"strongest_plasmon_remote_hotspot.svg")
    ids=[launch,geometry_remote,plasmon_remote_id];fig,ax=plt.subplots();vals=np.array([[m2i[x],m2s[x],m2[x]] for x in ids]);
    for col,label in enumerate(("incident","scattered","total")):ax.bar(np.arange(3)+(col-1)*.25,vals[:,col],.25,label=label)
    ax.set_yscale("log");ax.set_xticks(range(3),["source","geometric remote","plasmon remote"]);ax.legend();_save(fig,out/"remote_hotspot_field_decomposition.svg")
    strongest=transitions[int(np.argmax(ratios))];summary={"launch_hotspot":launch,"maximum_corrected_geodesic_nm":distance.max(),"zero_distance_hotspots":np.sum(distance==0),"strongest_ratio":strongest[8],"strongest_delta_log10":strongest[9],"strongest_upstream":strongest[0],"strongest_downstream":strongest[1],"strongest_upstream_distance_nm":strongest[2],"strongest_downstream_distance_nm":strongest[3],"strongest_downstream_position_nm":h.positions_nm[strongest[1]].tolist(),"strongest_gaps_nm":[strongest[14],strongest[15]],"strongest_coordination":[strongest[12],strongest[13]],"geometric_remote_hotspot":geometry_remote,"plasmon_remote_hotspot":plasmon_remote_id,"plasmon_remote_particles":h.pairs[q].tolist(),"plasmon_remote_position_nm":h.positions_nm[q].tolist(),"plasmon_remote_gap_nm":h.gaps_nm[q],"plasmon_remote_euclidean_nm":euclid[q],"plasmon_remote_geodesic_nm":distance[q],"plasmon_remote_steps":steps[q],"plasmon_remote_M2_inc":m2i[q],"plasmon_remote_M2_scat":m2s[q],"plasmon_remote_M2_total":m2[q],"plasmon_remote_dominance":dominance[q],"plasmon_remote_dominance_ge_1000":dominance[q]>=1000,"plasmon_remote_M4":m4[q],"plasmon_remote_source_M4_ratio":m4[q]/m4[launch],"plasmon_remote_coordination":a.degrees[h.pairs[q]].tolist(),"poynting_map":"not generated: magnetic-field reconstruction is not implemented"}
    for row in summaries:summary.update({f"{row[0]}_events":row[2],f"{row[0]}_probability":row[3],f"{row[0]}_remote_events":row[8],f"{row[0]}_remote_probability":row[9]})
    _write(data/"summary.csv",["metric","value"],summary.items());return summary
if __name__=="__main__":
    for k,v in run().items():print(f"{k}: {v}")
