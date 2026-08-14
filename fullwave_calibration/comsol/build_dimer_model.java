// GENERATED BUT NOT EXECUTED/VERIFIED IN COMSOL.
// COMSOL Java API template; verify feature identifiers for the installed release.
import com.comsol.model.*;
import com.comsol.model.util.*;
public class build_dimer_model {
  public static Model run() {
    Model m=ModelUtil.create("Dimer3D"); m.component().create("comp1",true);
    m.component("comp1").geom().create("geom1",3);
    String[][] p={{"R","10[nm]"},{"g","1[nm]"},{"lambda0","633[nm]"},
      {"theta","0[deg]"},{"E0","1[V/m]"},{"nm","1"},{"epsAg","-15+i*1"}};
    for(String[] x:p)m.param().set(x[0],x[1]);
    for(int q=1;q<=2;q++){String id="sph"+q;m.component("comp1").geom("geom1").create(id,"Sphere");
      m.component("comp1").geom("geom1").feature(id).set("r","R");
      m.component("comp1").geom("geom1").feature(id).set("pos",new String[]{q==1?"-(R+g/2)":"R+g/2","0","0"});}
    // Add surrounding sphere, PML selection, Ag/background materials, EWFD
    // scattered-field background E0*(cos(theta),sin(theta),0), three gap-refined
    // meshes, two domain/PML sizes, parametric sweeps, point evaluation at
    // (0,0,0), convergence checks, and CSV export per README_COMSOL.md.
    m.component("comp1").geom("geom1").run(); return m;
  }
  public static void main(String[] args){ModelUtil.initStandalone(true);run().save("dimer3d.mph");}
}
