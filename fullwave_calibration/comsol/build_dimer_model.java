import com.comsol.model.*;
import com.comsol.model.util.*;

/** Build the parameterized 3D Ag dimer geometry for COMSOL Multiphysics 6.x. */
public class build_dimer_model {
  public static Model run() {
    Model model = ModelUtil.create("Dimer633");
    model.modelPath("."); model.label("Ag_dimer_633nm.mph");
    model.param().set("R", "10[nm]"); model.param().set("gap", "3[nm]");
    model.param().set("lambda0", "633[nm]"); model.param().set("E0", "1[V/m]");
    model.param().set("pol", "0[deg]"); model.param().set("rdom", "400[nm]");
    model.param().set("tpml", "200[nm]");
    model.component().create("comp1", true); model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").lengthUnit("nm");
    model.component("comp1").geom("geom1").create("s1", "Sphere");
    model.component("comp1").geom("geom1").feature("s1").set("r", "R");
    model.component("comp1").geom("geom1").feature("s1").set("pos", new String[]{"-(R+gap/2)","0","0"});
    model.component("comp1").geom("geom1").create("s2", "Sphere");
    model.component("comp1").geom("geom1").feature("s2").set("r", "R");
    model.component("comp1").geom("geom1").feature("s2").set("pos", new String[]{"R+gap/2","0","0"});
    model.component("comp1").geom("geom1").create("air", "Sphere");
    model.component("comp1").geom("geom1").feature("air").set("r", "rdom");
    model.component("comp1").geom("geom1").create("outer", "Sphere");
    model.component("comp1").geom("geom1").feature("outer").set("r", "rdom+tpml");
    model.component("comp1").geom("geom1").run();
    // Domain selections, materials, PML, EWFD scattered-field background,
    // mesh/convergence studies, and exports are specified in the companion
    // document because their generated Java tags are release/license specific.
    model.save("Ag_dimer_633nm.mph"); return model;
  }
  public static void main(String[] args) { run(); ModelUtil.disconnect(); }
}
