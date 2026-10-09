"""Build water scenes around the collision-only, attributed upstream robot."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets" / "jumper"


@dataclass(frozen=True)
class Water:
    density: float = 1000.0  # kg/m³
    viscosity: float = 0.001  # Pa·s
    blunt_drag: float = 0.5
    slender_drag: float = 0.25
    angular_drag: float = 1.5
    kutta_lift: float = 1.0
    magnus_lift: float = 1.0
    buoyancy_ratio: float = 1.0  # displaced mass / link mass at reference density
    buoyancy_height_m: float = 0.01  # combined buoyancy center above COM, along torso +Z
    current: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world m/s

    def __post_init__(self):
        coefficients = (
            self.density,
            self.viscosity,
            self.blunt_drag,
            self.slender_drag,
            self.angular_drag,
            self.kutta_lift,
            self.magnus_lift,
            self.buoyancy_ratio,
            self.buoyancy_height_m,
        )
        if not np.isfinite(coefficients).all() or min(coefficients) < 0:
            raise ValueError("Water coefficients must be finite and nonnegative")
        if len(self.current) != 3 or not np.isfinite(self.current).all():
            raise ValueError("current must contain three finite velocities")


def make_model(mode: str = "planar", water: Water | None = None) -> mujoco.MjModel:
    if mode not in ("planar", "free", "tethered"):
        raise ValueError("mode must be 'planar', 'free' or 'tethered'")
    water = water or Water()
    root = ET.parse(ASSETS / "robot.xml").getroot()
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="buoyancy_height_m", data=str(water.buoyancy_height_m))
    root.find("compiler").set("meshdir", str(ASSETS))
    option = root.find("option")
    option.set("density", str(water.density))
    option.set("viscosity", str(water.viscosity))
    option.set("wind", " ".join(map(str, water.current)))
    # Fully submerged V_i = ratio * mass_i / 1000, at each link COM.
    # Model-owned compensation survives viewer.sync() (which clears external
    # perturbation forces) and mj_resetData. Gravity itself remains enabled.
    for body in root.findall(".//worldbody//body"):
        body.set("gravcomp", str(water.density * water.buoyancy_ratio / 1000.0))
    coefficients = " ".join(
        map(
            str,
            (
                water.blunt_drag,
                water.slender_drag,
                water.angular_drag,
                water.kutta_lift,
                water.magnus_lift,
            ),
        )
    )
    for geom in root.findall(".//worldbody//geom"):
        geom.set("fluidshape", "ellipsoid")
        geom.set("fluidcoef", coefficients)
    base = root.find(".//body[@name='base_link']")
    base.set("pos", "0 0 1")
    if mode == "planar":
        base.remove(base.find("joint[@name='floating_base']"))
        # Ideal fixed-depth, roll/pitch-constrained curriculum; still free to turn.
        for name, kind, axis in (
            ("root_x", "slide", "1 0 0"),
            ("root_y", "slide", "0 1 0"),
            ("root_yaw", "hinge", "0 0 1"),
        ):
            ET.SubElement(base, "joint", name=name, type=kind, axis=axis, limited="false")
    if mode == "tethered":
        equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name="thrust_stand",
            body1="base_link",
            solref="0.002 1",
            solimp="0.9999 0.9999 0.001",
        )
    actuator = ET.SubElement(root, "actuator")
    for joint in root.findall(".//worldbody//joint"):
        if joint.get("class") == "motor":
            ET.SubElement(
                actuator,
                "position",
                name=joint.get("name") + "_servo",
                joint=joint.get("name"),
                kp="3",
                kv="0.08",
                ctrlrange=joint.get("range"),
                forcerange="-2 2",
            )
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0.035 0.10 0.14",
        rgb2="0.055 0.15 0.19",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor",
        texture="grid",
        texrepeat="12 12",
        texuniform="true",
        reflectance="0.1",
    )
    world = root.find("worldbody")
    # A non-colliding visual reference. There is no surface/bottom propulsion.
    ET.SubElement(
        world,
        "geom",
        name="visual_floor",
        type="plane",
        size="6 6 0.1",
        pos="0 0 0",
        material="floor",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.9 1")
    ET.SubElement(world, "light", pos="2 -2 3", dir="-1 1 -1", diffuse="0.5 0.6 0.7")
    ET.SubElement(
        world,
        "geom",
        type="capsule",
        fromto="0 0 0.01 1 0 0.01",
        size="0.008",
        rgba="0.2 0.9 0.7 1",
        contype="0",
        conaffinity="0",
    )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def buoyancy_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World-frame restoring couple r_CB × F_b, with no net translation force.

    Link COM gravcomp supplies neutral lift. Moving the combined buoyancy center
    along torso +Z adds this conservative couple: U = -F_b · r_CB. It does not
    hold heading, damp angular velocity, or command any actuator.
    """
    base = model.body("base_link").id
    up = data.xmat[base].reshape(3, 3)[:, 2]
    height = model.numeric("buoyancy_height_m").data[0]
    displaced_mass = np.dot(model.body_mass, model.body_gravcomp)
    return np.cross(height * up, -displaced_mass * model.opt.gravity)


def step_water(model: mujoco.MjModel, data: mujoco.MjData, nstep: int = 1, *, measure_work=False):
    """Advance submerged dynamics; recompute the buoyancy couple every substep.

    Use this instead of mj_step for water scenes. External viewer perturbations
    are preserved, and no buoyancy force is left in viewer-owned xfrc_applied.
    Split stepping updates kinematics before calculating the couple and supports
    the scene's implicitfast integrator (RK4 is intentionally unsupported).
    With measure_work=True, return integrated absolute actuator work in joules.
    """
    if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("step_water requires Euler, implicit, or implicitfast")
    base = model.body("base_link").id
    original = data.xfrc_applied[base, 3:].copy()
    work = 0.0
    try:
        for _ in range(nstep):
            mujoco.mj_step1(model, data)
            data.xfrc_applied[base, 3:] = original + buoyancy_torque(model, data)
            mujoco.mj_step2(model, data)
            if measure_work:
                work += (
                    float(np.abs(data.actuator_force * data.actuator_velocity).sum())
                    * model.opt.timestep
                )
    finally:
        data.xfrc_applied[base, 3:] = original
    return work


def root_fluid_wrench(data: mujoco.MjData) -> np.ndarray:
    """Fluid-only wrench on a free root, excluding intrinsic COM buoyancy.

    The root has no joint spring/damper. Subtracting gravcomp works on MuJoCo
    3.3 as well as newer versions with a separate qfrc_fluid field.
    """
    return data.qfrc_passive[:6] - data.qfrc_gravcomp[:6]
