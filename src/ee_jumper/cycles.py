"""Periodic strokes, hydrodynamic impulse measurement and servo validation.

Impulse is the integral of external FLUID force, never tether reaction or peak
acceleration. Efficiency here is N s / J of absolute mechanical actuator work,
not a dimensionless propulsive or electrical efficiency.
"""

from dataclasses import asdict

import mujoco
import numpy as np

from ee_jumper.model import Water, make_model, root_fluid_wrench, step_water

DIRECTIONS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}
N_CYCLE_PARAMETERS = 36


class StrokeSpace:
    """18 moving joints, 2 coefficients each; fingers and wrists hold home.

    A warped phase makes position, velocity AND acceleration match at the
    common home pose every period. Different directions can switch there.
    """

    def __init__(self, model, period=2.0, max_excursion=0.9):
        if not np.isfinite(period) or period < 0.5:
            raise ValueError("period must be finite and >= 0.5 s")
        if not np.isfinite(max_excursion) or not 0 < max_excursion <= 2.0:
            raise ValueError("max_excursion must be in (0, 2] rad")
        self.max_excursion = float(max_excursion)
        self.period = float(period)
        joints = model.actuator_trnid[:, 0]
        self.qadr, self.vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.names = [model.joint(int(j)).name for j in joints]
        self.active = np.array(
            [i for i, n in enumerate(self.names) if int(n.split("_J")[1][0]) <= 2]
        )
        self.home = np.zeros(model.nu)
        for name, angle in {
            "LF_J0_joint": -0.75,
            "RF_J0_joint": 0.75,
            "LR_J0_joint": 0.75,
            "RR_J0_joint": -0.75,
            "LF_J4_joint": -0.6,
            "RF_J4_joint": 0.6,
        }.items():
            self.home[self.names.index(name)] = angle
        limits = model.jnt_range[joints]
        self.scale = np.minimum(
            max_excursion, 0.9 * np.minimum(self.home - limits[:, 0], limits[:, 1] - self.home)
        )
        self.scale /= 0.5 + np.sqrt(1.25)

    def trajectory(self, phases, parameters):
        p = np.asarray(parameters, dtype=float)
        if (
            p.shape != (N_CYCLE_PARAMETERS,)
            or not np.isfinite(p).all()
            or np.max(abs(p)) > 1.000001
        ):
            raise ValueError("Expected 36 finite stroke parameters in [-1, 1]")
        phases = np.atleast_1d(phases)
        if not np.isfinite(phases).all():
            raise ValueError("phases must be finite")
        z = 2 * np.pi * phases[:, None]
        phi = z - np.sin(z)
        omega = 2 * np.pi / self.period
        dphi, ddphi = omega * (1 - np.cos(z)), omega**2 * np.sin(z)
        a, b = p.reshape(-1, 2).T
        scale = self.scale[self.active]
        q = np.tile(self.home, (len(phases), 1))
        v, acc = np.zeros_like(q), np.zeros_like(q)
        q[:, self.active] += scale * (0.5 * a * (1 - np.cos(phi)) + b * np.sin(phi))
        tangent = 0.5 * a * np.sin(phi) + b * np.cos(phi)
        v[:, self.active] = scale * tangent * dphi
        acc[:, self.active] = scale * (
            (0.5 * a * np.cos(phi) - b * np.sin(phi)) * dphi**2 + tangent * ddphi
        )
        return q, v, acc


def summarize_cycle(force, torque, power, period, **extra):
    force, torque, power = np.asarray(force), np.asarray(torque), np.asarray(power)
    return {
        "mean_force_n": np.mean(force, axis=0).tolist(),
        "impulse_ns": (np.mean(force, axis=0) * period).tolist(),
        "mean_torque_nm": np.mean(torque, axis=0).tolist(),
        "angular_impulse_nms": (np.mean(torque, axis=0) * period).tolist(),
        "work_j": float(np.mean(power) * period),
        "mean_power_w": float(np.mean(power)),
        "period_s": period,
        **extra,
    }


def quality(metrics, direction, minimum_force=0.02, objective="efficiency"):
    """A minimum useful thrust stops vanishing-amplitude efficiency solutions."""
    if objective not in ("efficiency", "speed"):
        raise ValueError("objective must be efficiency or speed")
    d = DIRECTIONS[direction]
    force = np.asarray(metrics["mean_force_n"])
    axial = float(force @ d)
    lateral = float(np.linalg.norm(force - axial * d))
    moment = float(np.linalg.norm(metrics["mean_torque_nm"]))
    power = metrics["mean_power_w"]
    efficiency = axial / max(power, 1e-9)
    # Equivalent off-axis force also penalizes mean rotation, using a stated
    # 0.1 m reference length. No arbitrary electrical idle-power assumption.
    useful = axial - 1.5 * lateral - 2.0 * moment / 0.1
    physical = (
        metrics.get("peak_actuator_nm", 0) <= 2.00001
        and metrics.get("penetration_m", 0) <= 0.0005
        and all(
            c["q_closure_rad"] < 0.01 and c["v_closure_rad_s"] < 0.03
            for c in metrics.get("measured_cycles", [])
        )
        and not metrics.get("unstable", False)
    )
    feasible = physical and axial >= minimum_force and lateral <= 0.35 * axial and moment <= 0.002
    if not physical:
        score = (
            -100
            - min(metrics.get("peak_actuator_nm", 0), 1e6) / 2
            - metrics.get("penetration_m", 0) * 1000
        )
    elif feasible:
        # Thrust is a cheap speed surrogate on the stationary bench. Speed
        # training selects its final winner using measured free-swim velocity.
        score = 10 + (efficiency if objective == "efficiency" else (axial - 0.02 * power) / 0.1)
    elif objective == "speed":
        # High thrust must not flatten all infeasible candidates to the same
        # score: servo lag can add torque, and refinement needs a gradient
        # back toward each constraint before competing on speed.
        violation = (
            max(0, minimum_force - axial) / minimum_force
            + max(0, lateral - 0.35 * max(axial, 0)) / (0.35 * max(axial, minimum_force))
            + max(0, moment - 0.002) / 0.002
        )
        score = 9 - min(violation, 99) + 0.001 * np.clip(useful / minimum_force, -100, 100)
    else:
        score = min(9.0, useful / minimum_force)
    return {
        "axial_force_n": axial,
        "lateral_force_n": lateral,
        "torque_norm_nm": moment,
        "impulse_per_work_ns_per_j": efficiency,
        "feasible": bool(feasible),
        "score": float(score),
    }


def speed_quality(metrics, direction):
    """Free-swim ranking: progress first, modest work/drift/rotation costs.

    Velocity is COM displacement over the measurement window in the initial
    world axes. Mechanical power has weight 0.002 (m/s)/W, not an efficiency
    ratio that rewards arbitrarily slow motion.
    """
    velocity = np.asarray(metrics.get("mean_velocity_mps", [0, 0, 0]))
    axial = float(velocity @ DIRECTIONS[direction])
    lateral = float(np.linalg.norm(velocity - axial * DIRECTIONS[direction]))
    rotation = metrics.get("peak_rotation_rad", np.pi)
    valid = (
        not metrics.get("unstable", False)
        and axial > 0
        and rotation < 0.5
        and metrics.get("penetration_m", 0) <= 0.0005
        and metrics.get("peak_actuator_nm", 0) <= 2.00001
    )
    score = axial - 0.25 * lateral - 0.002 * metrics["mean_power_w"] - 0.005 * rotation
    return {
        "axial_speed_mps": axial,
        "lateral_speed_mps": lateral,
        "valid": bool(valid),
        "score": float(score if valid else -1 + min(axial, 0)),
    }


class KinematicBench:
    """Fast periodic inverse-dynamics search, followed by a dynamic bench.

    Gravity=0 is equivalent to the default per-link neutral COM buoyancy.
    The free root is prescribed stationary, preserving its force projection.
    Root fluid force excludes gravity compensation; no root spring/damper exists.
    """

    def __init__(self, period=2.0, samples=96):
        if samples < 16:
            raise ValueError("Need at least 16 integration samples per cycle")
        self.model = make_model("free")
        self.model.opt.gravity[:] = 0
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        self.data = mujoco.MjData(self.model)
        self.stroke = StrokeSpace(self.model, period)
        self.samples = samples

    def evaluate(self, parameters):
        m, d, s = self.model, self.data, self.stroke
        mujoco.mj_resetData(m, d)
        q, v, acc = s.trajectory((np.arange(self.samples) + 0.5) / self.samples, parameters)
        forces, moments, powers = [], [], []
        peak, penetration = 0.0, 0.0
        for i in range(self.samples):
            d.qpos[s.qadr], d.qvel[s.vadr], d.qacc[s.vadr] = q[i], v[i], acc[i]
            mujoco.mj_inverse(m, d)
            wrench = root_fluid_wrench(d)
            forces.append(wrench[:3])
            moments.append(wrench[3:6])
            tau = d.qfrc_inverse[s.vadr]
            powers.append(float(np.abs(tau * v[i]).sum()))
            peak = max(peak, float(np.max(np.abs(tau))))
            if d.ncon:
                penetration = max(penetration, -float(np.min(d.contact.dist[: d.ncon])))
        return summarize_cycle(
            forces,
            moments,
            powers,
            s.period,
            peak_actuator_nm=peak,
            penetration_m=penetration,
            method="prescribed_periodic_inverse_dynamics",
        )


class DynamicBench:
    def __init__(self, period=2.0, control_dt=0.01, mode="tethered", max_excursion=0.9):
        if mode not in ("tethered", "free"):
            raise ValueError("Dynamic bench mode must be tethered or free")
        self.mode, self.water = mode, Water()
        self.model = make_model(mode)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        self.data = mujoco.MjData(self.model)
        self.stroke = StrokeSpace(self.model, period, max_excursion)
        self.substeps = round(control_dt / self.model.opt.timestep)
        self.steps_per_cycle = round(period / control_dt)
        if (
            self.substeps < 1
            or not np.isclose(self.substeps * self.model.opt.timestep, control_dt)
            or not np.isclose(self.steps_per_cycle * control_dt, period)
        ):
            raise ValueError("period/control_dt/timestep must divide exactly")
        self.dt = control_dt

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.stroke.qadr] = self.stroke.home
        self.data.ctrl[:] = self.stroke.home
        mujoco.mj_forward(self.model, self.data)

    def evaluate(self, parameters, warmup_cycles=3, measurement_cycles=2, callback=None):
        if warmup_cycles < 0 or measurement_cycles < 1:
            raise ValueError("Invalid cycle counts")
        self.reset()
        m, d, s = self.model, self.data, self.stroke
        targets, _, _ = s.trajectory(
            np.arange(self.steps_per_cycle) / self.steps_per_cycle, parameters
        )
        forces, moments, powers, cycles = [], [], [], []
        peak, penetration, tracking, base_motion = 0.0, 0.0, 0.0, 0.0
        start_com = d.subtree_com[m.body("base_link").id].copy()
        measurement_start_com = start_com.copy()
        rotation_angle = 0.0
        q_low, q_high = d.qpos[s.qadr].copy(), d.qpos[s.qadr].copy()
        unstable = False
        for cycle in range(warmup_cycles + measurement_cycles):
            if cycle == warmup_cycles:
                mujoco.mj_forward(m, d)
                measurement_start_com = d.subtree_com[m.body("base_link").id].copy()
            start_q, start_v = d.qpos[s.qadr].copy(), d.qvel[s.vadr].copy()
            begin = len(forces)
            for step, target in enumerate(targets):
                d.ctrl[:] = target
                expected = d.time + self.dt
                step_water(m, d, nstep=self.substeps)
                if (
                    d.time < expected - 1e-8
                    or not np.isfinite(d.qpos).all()
                    or not np.isfinite(d.qvel).all()
                ):
                    unstable = True
                    break
                if cycle >= warmup_cycles:
                    # qfrc_passive and actuator_velocity/force are from the same
                    # final 1 ms substep, avoiding a mixed-time power estimate.
                    wrench = root_fluid_wrench(d)
                    forces.append(wrench[:3])
                    moments.append(wrench[3:6])
                    powers.append(float(np.abs(d.actuator_force * d.actuator_velocity).sum()))
                    peak = max(peak, float(np.max(np.abs(d.actuator_force))))
                    tracking = max(tracking, float(np.max(abs(d.qpos[s.qadr] - target))))
                    if d.ncon:
                        penetration = max(penetration, -float(np.min(d.contact.dist[: d.ncon])))
                q_low = np.minimum(q_low, d.qpos[s.qadr])
                q_high = np.maximum(q_high, d.qpos[s.qadr])
                base_motion = max(base_motion, float(np.linalg.norm(d.qpos[:3] - m.qpos0[:3])))
                rotation_angle = max(
                    rotation_angle, float(2 * np.arccos(np.clip(abs(d.qpos[3]), 0, 1)))
                )
                if callback is not None:
                    callback(cycle * self.steps_per_cycle + step, self)
            if unstable:
                break
            if cycle >= warmup_cycles:
                cycles.append(
                    {
                        **summarize_cycle(
                            forces[begin:], moments[begin:], powers[begin:], s.period
                        ),
                        "q_closure_rad": float(np.max(abs(d.qpos[s.qadr] - start_q))),
                        "v_closure_rad_s": float(np.max(abs(d.qvel[s.vadr] - start_v))),
                    }
                )
        mujoco.mj_forward(m, d)
        if not forces or unstable:
            return summarize_cycle(
                [[0, 0, 0]],
                [[0, 0, 0]],
                [0],
                s.period,
                unstable=True,
                peak_actuator_nm=peak,
                penetration_m=penetration,
            )
        return summarize_cycle(
            forces,
            moments,
            powers,
            s.period,
            method=f"dynamic_{self.mode}",
            warmup_cycles=warmup_cycles,
            measured_cycles=cycles,
            peak_actuator_nm=peak,
            penetration_m=penetration,
            peak_tracking_error_rad=tracking,
            max_base_displacement_m=base_motion,
            unstable=unstable,
            displacement_m=(d.subtree_com[m.body("base_link").id] - start_com).tolist(),
            measurement_displacement_m=(
                d.subtree_com[m.body("base_link").id] - measurement_start_com
            ).tolist(),
            mean_velocity_mps=(
                (d.subtree_com[m.body("base_link").id] - measurement_start_com)
                / (measurement_cycles * s.period)
            ).tolist(),
            measurement_seconds=measurement_cycles * s.period,
            peak_rotation_rad=rotation_angle,
            actual_joint_peak_to_peak_rad=(q_high - q_low).tolist(),
            final_quaternion_wxyz=d.qpos[3:7].tolist(),
            elapsed_s=float(d.time),
            water=asdict(self.water),
        )
