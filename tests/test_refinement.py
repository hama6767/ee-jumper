"""Motion export must preserve a working baseline and stay inside servo limits."""

import json
from pathlib import Path

import numpy as np
import pytest

from ee_jumper.cycles import DIRECTIONS, StrokeSpace
from ee_jumper.learn_cycles import load_library
from ee_jumper.model import make_model
from ee_jumper.refine_ppo import MotionRefinementEnv, candidate_library
from ee_jumper.simulation import Swimmer


@pytest.fixture(scope="module")
def refinement():
    source = Path(__file__).resolve().parents[1] / "examples/strokes-xyz-cem.json"
    return MotionRefinementEnv(load_library(source))


def test_range_extension_leaves_wrist_fingers_and_torque_unchanged(refinement):
    old = make_model("free")
    new = refinement.bench.model
    active = refinement.bench.stroke.active
    delta = new.actuator_ctrlrange - old.actuator_ctrlrange
    np.testing.assert_allclose(delta[active], np.tile(np.deg2rad([-10, 10]), (18, 1)))
    fixed = np.setdiff1d(np.arange(old.nu), active)
    np.testing.assert_allclose(delta[fixed], 0)
    np.testing.assert_array_equal(old.actuator_forcerange, new.actuator_forcerange)
    np.testing.assert_array_equal(new.jnt_range[new.actuator_trnid[:, 0]], new.actuator_ctrlrange)


def test_zero_ppo_action_preserves_every_original_joint_trajectory(refinement):
    old = StrokeSpace(make_model("free"), 1, 1.6)
    phases = np.linspace(0, 1, 200)
    exported = candidate_library(refinement.library, refinement, np.zeros((6, 18)))
    runtime = Swimmer(exported)
    for name in DIRECTIONS:
        original = old.trajectory(phases, refinement.library["strokes"][name]["parameters"])
        replayed = runtime.controller.stroke.trajectory(
            phases, exported["strokes"][name]["parameters"]
        )
        np.testing.assert_allclose(replayed, original, atol=1e-12)
        # Exploration cannot break periodic boundaries or exceed joint limits.
        p = refinement.parameters(name, np.linspace(-1, 1, 18))
        stroke = runtime.controller.stroke
        q = stroke.trajectory(phases, p)[0]
        assert np.all(q >= runtime.model.actuator_ctrlrange[:, 0])
        assert np.all(q <= runtime.model.actuator_ctrlrange[:, 1])
        q, v, a = stroke.trajectory([0, 1], p)
        np.testing.assert_allclose(q, np.tile(stroke.home, (2, 1)), atol=1e-12)
        np.testing.assert_allclose(v, 0, atol=1e-12)
        np.testing.assert_allclose(a, 0, atol=1e-12)


def test_ppo_trial_rewards_baseline_zero_and_requires_reset(refinement):
    obs, _ = refinement.reset(seed=19)
    assert refinement.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = refinement.step(np.zeros(18))
    assert refinement.observation_space.contains(obs)
    assert reward == pytest.approx(0, abs=1e-9)
    assert terminated and not truncated and info["valid"]
    with pytest.raises(RuntimeError, match="reset"):
        refinement.step(np.zeros(18))
    with pytest.raises(ValueError, match="finite corrections"):
        refinement.parameters("+x", np.full(18, np.nan))


def test_unmeasured_ppo_candidate_cannot_be_used_as_validated_library(refinement, tmp_path):
    candidate = candidate_library(refinement.library, refinement, np.zeros((6, 18)))
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="Unvalidated"):
        load_library(path)
    assert load_library(path, require_feasible=False)["validation_pending"]


@pytest.mark.parametrize("margin", [-0.1, float("nan"), 0.4])
def test_invalid_range_extension_rejected(refinement, tmp_path, margin):
    with pytest.raises(ValueError, match="joint_limit_margin"):
        make_model("free", joint_limit_margin_rad=margin)
    source = dict(refinement.library, joint_limit_margin_rad=margin)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="joint limit margin"):
        load_library(path)
