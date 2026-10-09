import hashlib
import json

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import ee_jumper  # noqa: F401
from ee_jumper.env import SwimEnv
from ee_jumper.gait import action_at, rollout
from ee_jumper.model import (
    ASSETS,
    Water,
    buoyancy_torque,
    make_model,
    root_fluid_wrench,
    step_water,
)
from ee_jumper.train import load_gait, train_cem


def test_vendored_assets_match_provenance():
    provenance = json.loads((ASSETS / "provenance.json").read_text())
    assert (
        hashlib.sha256((ASSETS / "robot.xml").read_bytes()).hexdigest()
        == provenance["derived_xml_sha256"]
    )
    for record in provenance["meshes"]:
        assert (
            hashlib.sha256((ASSETS / record["local"]).read_bytes()).hexdigest() == record["sha256"]
        )


@pytest.mark.parametrize("mode,nv", [("planar", 25), ("free", 28)])
def test_model_and_gym_contract(mode, nv):
    env = SwimEnv(mode=mode, episode_seconds=0.2)
    try:
        assert env.model.nu == 22 and env.model.nv == nv
        assert np.sum(env.model.body_mass) == pytest.approx(2.54296197)
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=42)
        assert env.observation_space.contains(observation)
        for i in range(10):
            obs, _, terminated, truncated, _ = env.step(np.zeros(22))
            assert np.isfinite(obs).all() and not terminated
            assert truncated == (i == 9)
        with pytest.raises(RuntimeError):
            env.step(np.zeros(22))
    finally:
        env.close()


def test_registration():
    env = gym.make("EEJumperSwim-v0")
    env.reset(seed=0)
    assert env.action_space.shape == (22,)
    env.close()


def test_neutral_buoyancy_balances_gravity():
    water = Water()
    model = make_model("free", water)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qacc, 0, atol=1e-9)
    np.testing.assert_allclose(data.qfrc_gravcomp[:3], -model.body_mass.sum() * model.opt.gravity)
    np.testing.assert_allclose(root_fluid_wrench(data), 0, atol=1e-12)


def test_subneutral_buoyancy_sinks():
    model = make_model("free", Water(buoyancy_ratio=0.9))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.qacc[2] < -0.5


def test_zero_action_starts_at_rest_without_reset_noise():
    env = SwimEnv(mode="free", episode_seconds=1)
    try:
        env.reset()
        initial = env.data.qpos.copy()
        for _ in range(50):
            env.step(np.zeros(22))
        np.testing.assert_allclose(env.data.qpos, initial, atol=1e-10)
        np.testing.assert_allclose(env.data.qvel, 0, atol=1e-10)
    finally:
        env.close()


def test_fluid_drag_opposes_translation_and_current_is_relative():
    for speed in (-0.3, 0.3):
        model = make_model("free", Water(kutta_lift=0, magnus_lift=0))
        data = mujoco.MjData(model)
        data.qvel[0] = speed
        mujoco.mj_forward(model, data)
        assert data.qfrc_passive[0] * speed < 0  # dissipates energy
    model = make_model("free", Water(current=(0.3, 0, 0)))
    data = mujoco.MjData(model)
    data.qvel[0] = 0.3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(root_fluid_wrench(data), 0, atol=1e-10)


def test_lift_coefficients_affect_oblique_flow():
    forces = []
    for lift in (0.0, 1.0):
        model = make_model("free", Water(kutta_lift=lift, magnus_lift=lift))
        data = mujoco.MjData(model)
        data.qvel[:6] = [0.3, 0.2, 0.1, 0.4, -0.2, 0.3]
        mujoco.mj_forward(model, data)
        forces.append(data.qfrc_passive.copy())
    assert np.linalg.norm(forces[1] - forces[0]) > 1e-4


def test_joint_motion_in_vacuum_cannot_generate_forward_momentum():
    env = SwimEnv(water=Water(density=0, viscosity=0), episode_seconds=1.0)
    # Neutralize gravity to isolate conservation in an otherwise empty world.
    env.model.opt.gravity[:] = 0
    p = np.full(13, 0.4)
    try:
        result = rollout(env, p)
        assert abs(result["x_m"]) < 0.001
    finally:
        env.close()


def test_random_control_is_finite_and_actuators_are_limited():
    rng = np.random.default_rng(3)
    env = SwimEnv(episode_seconds=2)
    try:
        env.reset(seed=3)
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 22))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert np.max(np.abs(env.data.actuator_force)) <= 2.00001
            assert not info["unstable"]
            if terminated or truncated:
                break
    finally:
        env.close()


def test_invalid_action_and_configuration_rejected():
    with pytest.raises(ValueError):
        Water(density=-1)
    with pytest.raises(ValueError):
        SwimEnv(mode="typo")
    with pytest.raises(ValueError):
        SwimEnv(episode_seconds=0)
    env = SwimEnv()
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step(np.full(22, np.nan))
        with pytest.raises(ValueError):
            env.step(np.zeros(21))
        assert np.max(np.abs(action_at(0.2, np.ones(13), env.joint_names))) <= 1
    finally:
        env.close()


def test_instability_returns_finite_failure(monkeypatch):
    env = SwimEnv()
    try:
        env.reset(seed=1)

        def failed_step(model, data):
            data.qpos[:] = np.nan
            data.qvel[:] = np.nan

        monkeypatch.setattr("ee_jumper.env.step_water", failed_step)
        observation, reward, terminated, _, info = env.step(np.zeros(22))
        assert terminated and info["unstable"]
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert reward < -0.9
    finally:
        env.close()


def test_buoyancy_offset_is_restoring_and_does_not_control_yaw():
    model = make_model("free")
    data = mujoco.MjData(model)
    angle = np.deg2rad(25)
    data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    mujoco.mj_forward(model, data)
    torque = buoyancy_torque(model, data)
    expected = -model.body_mass.sum() * 9.81 * 0.01 * np.sin(angle)
    np.testing.assert_allclose(torque, [expected, 0, 0], atol=1e-12)
    # Torque is a pure couple and never persists in viewer perturbation storage.
    data.xfrc_applied[model.body("base_link").id, :3] = [0.1, 0.2, 0.3]
    original = data.xfrc_applied.copy()
    step_water(model, data, 5)
    np.testing.assert_array_equal(data.xfrc_applied, original)
    mujoco.mj_resetData(model, data)
    data.qpos[3:7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(buoyancy_torque(model, data), 0, atol=1e-12)


def test_passive_recovery_from_roll_and_pitch():
    from ee_jumper.cycles import DynamicBench

    bench = DynamicBench(mode="free")
    for axis in (1, 2):
        bench.reset()
        angle = np.deg2rad(25)
        bench.data.qpos[3:7] = [np.cos(angle / 2), 0, 0, 0]
        bench.data.qpos[3 + axis] = np.sin(angle / 2)
        for _ in range(1000):
            bench.data.xfrc_applied[:] = 0
            step_water(bench.model, bench.data, 10)
        mujoco.mj_forward(bench.model, bench.data)
        up = bench.data.xmat[bench.model.body("base_link").id].reshape(3, 3)[2, 2]
        assert np.arccos(np.clip(up, -1, 1)) < np.deg2rad(1)
        np.testing.assert_array_equal(bench.data.ctrl, bench.stroke.home)


def test_search_checkpoint_and_replay(tmp_path):
    path = tmp_path / "gait.json"
    result = train_cem(path, generations=1, population=4, seconds=0.1, seed=7)
    assert load_gait(path)["parameters"] == result["parameters"]
    assert len(result["evaluation"]) == 3
    env = SwimEnv(episode_seconds=0.1)
    try:
        replay = rollout(env, result["parameters"], seed=7)
        assert replay["return"] == pytest.approx(result["training_result"]["return"])
    finally:
        env.close()
