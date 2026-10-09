import json
from pathlib import Path

import numpy as np
import pytest

from ee_jumper.control import CycleController, project_duty
from ee_jumper.cycles import (
    DIRECTIONS,
    DynamicBench,
    KinematicBench,
    StrokeSpace,
    quality,
    speed_quality,
)
from ee_jumper.learn_cycles import load_library
from ee_jumper.model import make_model, step_water


@pytest.fixture(params=[0.9, 1.6])
def stroke(request):
    return StrokeSpace(make_model("free"), max_excursion=request.param)


def test_all_strokes_share_position_velocity_acceleration_at_boundaries(stroke):
    rng = np.random.default_rng(9)
    for _ in range(10):
        p = rng.uniform(-1, 1, 36)
        q, v, a = stroke.trajectory([0, 1], p)
        np.testing.assert_allclose(q, np.tile(stroke.home, (2, 1)), atol=1e-12)
        np.testing.assert_allclose(v, 0, atol=1e-12)
        np.testing.assert_allclose(a, 0, atol=1e-12)


def test_analytic_stroke_derivatives_and_joint_limits(stroke):
    p = np.random.default_rng(7).uniform(-1, 1, 36)
    phases = np.linspace(0, 1, 1000)
    q, v, a = stroke.trajectory(phases, p)
    step = stroke.period / 999
    np.testing.assert_allclose(np.gradient(q, step, axis=0)[1:-1], v[1:-1], atol=2e-4)
    np.testing.assert_allclose(np.gradient(v, step, axis=0)[1:-1], a[1:-1], atol=2e-3)
    model = make_model("free")
    limits = model.actuator_ctrlrange
    assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])


def test_fluid_impulse_is_not_inertial_reaction():
    bench = KinematicBench(samples=64)
    p = np.random.default_rng(4).normal(0, 0.3, 36).clip(-1, 1)
    wet = bench.evaluate(p)
    assert np.linalg.norm(wet["impulse_ns"]) > 0.02
    bench.model.opt.density = bench.model.opt.viscosity = 0
    dry = bench.evaluate(p)
    np.testing.assert_allclose(dry["impulse_ns"], 0, atol=1e-12)
    # Motors still accelerate moving limbs, but it is not counted as thrust.
    assert dry["work_j"] > 0


def test_zero_stroke_has_no_propulsion_or_efficiency_loophole():
    metrics = KinematicBench(samples=32).evaluate(np.zeros(36))
    np.testing.assert_allclose(metrics["impulse_ns"], 0, atol=1e-12)
    assert not quality(metrics, "+x")["feasible"]


def test_idle_survives_viewer_force_clearing_and_reset():
    bench = DynamicBench(period=1, mode="free")
    for _ in range(2):
        bench.reset()
        initial = bench.data.qpos.copy()
        for _ in range(200):
            bench.data.xfrc_applied[:] = 0
            step_water(bench.model, bench.data, nstep=10)
        np.testing.assert_allclose(bench.data.qpos, initial, atol=1e-10)
        np.testing.assert_allclose(bench.data.qvel, 0, atol=1e-10)
    measured = bench.evaluate(np.zeros(36), warmup_cycles=0, measurement_cycles=1)
    np.testing.assert_allclose(measured["impulse_ns"], 0, atol=1e-10)
    np.testing.assert_allclose(measured["mean_torque_nm"], 0, atol=1e-10)


def test_speed_objective_accepts_more_work_for_faster_motion():
    slow = {
        "mean_force_n": [0.04, 0, 0],
        "mean_torque_nm": [0, 0, 0],
        "mean_power_w": 0.04,
        "mean_velocity_mps": [0.01, 0, 0],
        "peak_rotation_rad": 0,
    }
    fast = {
        **slow,
        "mean_force_n": [0.12, 0, 0],
        "mean_power_w": 0.2,
        "mean_velocity_mps": [0.025, 0, 0],
    }
    assert quality(slow, "+x")["score"] > quality(fast, "+x")["score"]
    assert (
        quality(fast, "+x", objective="speed")["score"]
        > quality(slow, "+x", objective="speed")["score"]
    )
    assert speed_quality(fast, "+x")["score"] > speed_quality(slow, "+x")["score"]
    assert not speed_quality({**fast, "peak_rotation_rad": 0.6}, "+x")["valid"]
    # An infeasible high-thrust candidate must improve its torque rather than
    # get stuck on the old clipped-score plateau.
    far = {**fast, "mean_torque_nm": [0, 0.006, 0]}
    near = {**fast, "mean_torque_nm": [0, 0.003, 0]}
    assert (
        quality(near, "+x", objective="speed")["score"]
        > quality(far, "+x", objective="speed")["score"]
    )


def test_dynamic_cycles_settle_and_close():
    bench = DynamicBench(period=1, control_dt=0.02)
    p = np.random.default_rng(4).normal(0, 0.1, 36).clip(-1, 1)
    metrics = bench.evaluate(p, warmup_cycles=2, measurement_cycles=2)
    assert not metrics["unstable"]
    assert metrics["max_base_displacement_m"] < 1e-5
    assert metrics["peak_actuator_nm"] <= 2.00001
    for cycle in metrics["measured_cycles"]:
        assert cycle["q_closure_rad"] < 0.001
        assert cycle["v_closure_rad_s"] < 0.005


def test_constrained_efficiency_ranking():
    good = {"mean_force_n": [0.03, 0, 0], "mean_torque_nm": [0, 0, 0], "mean_power_w": 0.1}
    assert quality(good, "+x")["feasible"]
    efficient = {**good, "mean_power_w": 0.05}
    assert quality(efficient, "+x")["score"] > quality(good, "+x")["score"]
    weak = {**good, "mean_force_n": [0.001, 0, 0], "mean_power_w": 0.00001}
    assert quality(weak, "+x")["score"] < quality(good, "+x")["score"]
    assert not quality({**good, "mean_torque_nm": [0.01, 0, 0]}, "+x")["feasible"]
    assert not quality({**good, "peak_actuator_nm": 2.1}, "+x")["feasible"]


@pytest.fixture
def library():
    # Synthetic orthogonal calibration isolates scheduling from fluid physics.
    return {
        "format": "ee-jumper-strokes-v1",
        "period_s": 2.0,
        "minimum_force_n": 0.02,
        "strokes": {
            name: {
                "parameters": np.full(36, 0.1).tolist(),
                "metrics": {
                    "mean_force_n": (axis * 0.05).tolist(),
                    "mean_torque_nm": [0, 0, 0],
                    "mean_power_w": 0.1,
                },
            }
            for name, axis in DIRECTIONS.items()
        },
    }


def test_duty_allocator_bounds_and_mixed_axes(library):
    controller = CycleController(make_model("free"), library)
    expected = np.array([0.02, -0.01, 0.005])
    duty = controller.allocate(expected)
    np.testing.assert_allclose(controller.forces @ duty, expected, atol=1e-7)
    assert min(duty) >= 0 and sum(duty) <= 1 + 1e-12
    saturated = controller.allocate([1.0, 1.0, 1.0])
    assert min(saturated) >= 0 and sum(saturated) <= 1 + 1e-12
    np.testing.assert_allclose(project_duty(np.array([-3.0, 0, 2.0])), [0, 0, 1])


def test_free_speed_calibration_keeps_powerful_strokes_active(library):
    for name, axis in DIRECTIONS.items():
        record = library["strokes"][name]
        record["metrics"]["mean_force_n"] = (axis * 0.5).tolist()
        record["free_validation"] = {"displacement_m": (axis * 0.12).tolist(), "elapsed_s": 6}
    controller = CycleController(make_model("free"), library)
    controller.set_command([0.02, 0, 0], 0)
    result = controller.step(0.01, [0.02, 0, 0], 0)
    assert result.active_stroke == "+x" and result.duties[0] > 0.99
    controller.set_command([0, 0, 0], 0.01)
    np.testing.assert_allclose(controller.feedforward_force, 0)
    assert controller.step(0.01, [0.02, 0, 0], 0.01).active_stroke is None


def test_timeout_returns_home_and_direction_changes_only_at_boundary(library):
    controller = CycleController(make_model("free"), library, command_timeout=0.5)
    for i in range(20):
        controller.set_command([0.05, 0, 0], i * 0.02)
        output = controller.step(0.02, [0, 0, 0], i * 0.02)
        assert output.active_stroke == "+x"
    controller.set_command([-0.05, 0, 0], 0.4)
    output = controller.step(0.02, [0, 0, 0], 0.4)
    assert output.active_stroke == "+x"
    for i in range(100):
        output = controller.step(0.02, [0, 0, 0], 1 + i * 0.02)
    assert output.command_stale and output.active_stroke is None
    np.testing.assert_allclose(output.joint_positions, controller.stroke.home)


def test_library_requires_six_feasible_directions(tmp_path, library):
    path = tmp_path / "library.json"
    path.write_text(json.dumps(library))
    assert len(load_library(path)["strokes"]) == 6
    library["strokes"]["+y"]["metrics"]["mean_force_n"] = [0, 0, 0]
    path.write_text(json.dumps(library))
    with pytest.raises(ValueError, match="\\+y"):
        load_library(path)
    del library["strokes"]["+y"]
    path.write_text(json.dumps(library))
    with pytest.raises(ValueError, match="complete"):
        load_library(path)


def test_recorded_library_all_directions_and_free_motion():
    path = Path(__file__).resolve().parents[1] / "examples/strokes-xyz.json"
    library = load_library(path)
    for name, record in library["strokes"].items():
        metrics = record["metrics"]
        np.testing.assert_allclose(
            np.array(metrics["mean_force_n"]) * library["period_s"], metrics["impulse_ns"]
        )
        if library["format"] == "ee-jumper-strokes-v2":
            from ee_jumper.learn_displacement import displacement_quality

            assert displacement_quality(record["free_validation"], name)["valid"]
        else:
            assert quality(metrics, name, library["minimum_force_n"])["feasible"]
        assert np.array(record["free_validation"]["displacement_m"]) @ DIRECTIONS[name] > 0


def test_large_free_stroke_cannot_swim_without_fluid():
    bench = DynamicBench(period=1, mode="free", max_excursion=1.6)
    bench.model.opt.density = bench.model.opt.viscosity = 0
    result = bench.evaluate(np.random.default_rng(13).normal(0, 0.12, 36), 0, 2)
    assert np.linalg.norm(result["displacement_m"]) < 1e-4


def test_displacement_library_validates_motion_not_static_torque(tmp_path, library):
    library["format"] = "ee-jumper-strokes-v2"
    library["max_excursion_rad"] = 1.6
    for name, record in library["strokes"].items():
        record["metrics"]["mean_torque_nm"] = [0, 0.01, 0]
        record["free_validation"] = {
            "displacement_m": (DIRECTIONS[name] * 0.2).tolist(),
            "elapsed_s": 6,
            "peak_rotation_rad": 0.1,
            "peak_actuator_nm": 1.5,
            "penetration_m": 0,
            "work_j": 1,
        }
    path = tmp_path / "library.json"
    path.write_text(json.dumps(library))
    assert load_library(path)["max_excursion_rad"] == 1.6
    library["strokes"]["+x"]["free_validation"]["displacement_m"] = [0, 0.2, 0]
    path.write_text(json.dumps(library))
    with pytest.raises(ValueError, match="displacement"):
        load_library(path)


def test_keyboard_latches_axes_and_stops():
    from ee_jumper.keyboard import KeyboardCommand

    keys = KeyboardCommand(0.08)
    assert keys.press("W") and keys.press("r")
    assert keys.velocity == [0.08, 0, 0.08]
    keys.press("s")
    assert keys.velocity == [-0.08, 0, 0.08]
    assert keys.press("space") and keys.velocity == [0, 0, 0]
    assert not keys.press("q")
    with pytest.raises(ValueError):
        KeyboardCommand(float("nan"))
