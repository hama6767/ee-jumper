"""Gymnasium task: learn forward swimming in homogeneous, unbounded water."""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ee_jumper.model import Water, make_model, step_water


class SwimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        mode="planar",
        water=None,
        episode_seconds=10.0,
        render_mode=None,
        width=800,
        height=600,
    ):
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Unsupported render_mode")
        if not np.isfinite(episode_seconds) or episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive and finite")
        self.mode, self.water, self.render_mode = mode, water or Water(), render_mode
        self.model = make_model(mode, self.water)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 20
        self.dt = self.frame_skip * self.model.opt.timestep
        self.max_steps = int(np.ceil(episode_seconds / self.dt))
        self.base = self.model.body("base_link").id
        self.joints = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.joint_names = [self.model.joint(int(j)).name for j in self.joints]
        self.home = np.zeros(self.model.nu)
        for side, sign in (("LF", -1), ("RF", 1)):
            self.home[self.joint_names.index(f"{side}_J4_joint")] = sign * 0.6
        self.action_scale = 0.65
        self.previous_action = np.zeros(self.model.nu)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # Joint angles/velocities, body-frame velocity/omega, up vector,
        # world-forward in body frame, depth error, previous action.
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3 * self.model.nu + 13,), dtype=np.float32
        )
        self.width, self.height = width, height
        self.renderer = self.viewer = None
        self.steps = 0
        self.needs_reset = True

    def _observation(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 1
        )
        return np.concatenate(
            (
                self.data.qpos[self.qadr] - self.home,
                0.1 * self.data.qvel[self.vadr],
                velocity,
                rotation[2, :],
                rotation[0, :],
                [self.data.xpos[self.base, 2] - 1.0],
                self.previous_action,
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = 0.0 if options is None else float(options.get("noise", 0.0))
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("reset noise must be finite and nonnegative")
        self.data.qpos[self.qadr] = np.clip(
            self.home + self.np_random.uniform(-noise, noise, self.model.nu),
            self.model.jnt_range[self.joints, 0],
            self.model.jnt_range[self.joints, 1],
        )
        self.data.ctrl[:] = self.home
        self.previous_action[:] = 0
        self.steps, self.needs_reset = 0, False
        mujoco.mj_forward(self.model, self.data)
        self.start_com = self.data.subtree_com[self.base].copy()
        return self._observation(), {"mode": self.mode}

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError(f"Expected {self.model.nu} finite actions")
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = np.clip(
            self.home + self.action_scale * action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        before = self.data.subtree_com[self.base].copy()
        previous_qpos = self.data.qpos.copy()
        previous_time = self.data.time
        energy = 0.0
        unstable = False
        for _ in range(self.frame_skip):
            expected_time = self.data.time + self.model.opt.timestep
            step_water(self.model, self.data)
            energy += (
                float(np.abs(self.data.actuator_force * self.data.qvel[self.vadr]).sum())
                * self.model.opt.timestep
            )
            # MuJoCo may reset unstable state to a finite default; detect that too.
            if (
                self.data.time < expected_time - 1e-8
                or not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
            ):
                unstable = True
                break
        if unstable:
            # Return the last valid pose with a failure flag, never a NaN
            # observation or a spurious displacement caused by MuJoCo's reset.
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = previous_qpos
            self.data.time = previous_time
            if not np.isfinite(energy):
                energy = 0.0
        mujoco.mj_forward(self.model, self.data)  # sync pose/COM with integrated state
        displacement = self.data.subtree_com[self.base] - self.start_com
        velocity = (self.data.subtree_com[self.base] - before) / self.dt
        rotation = self.data.xmat[self.base].reshape(3, 3)
        upright = float(rotation[2, 2])
        depth_error = float(self.data.xpos[self.base, 2] - 1.0)
        smoothness = float(np.square(action - self.previous_action).mean())
        # Integrated reward, invariant to control rate. Progress uses whole-robot
        # COM so simply moving a limb relative to the torso cannot earn progress.
        reward = (
            float(velocity[0])
            - 0.5 * float(velocity[1] ** 2)
            - 0.1 * (1.0 - float(rotation[0, 0]))
            - 0.2 * (1.0 - upright)
            - 0.5 * depth_error**2
        ) * self.dt
        reward -= 0.002 * energy + 0.002 * smoothness
        self.steps += 1
        terminated = bool(
            unstable
            or upright < 0.3
            or abs(depth_error) > 0.7
            or np.linalg.norm(displacement[:2]) > 5.0
        )
        truncated = self.steps >= self.max_steps
        if terminated:
            reward -= 1.0
        self.previous_action[:] = action
        self.needs_reset = terminated or truncated
        info = {
            "x_m": float(displacement[0]),
            "y_m": float(displacement[1]),
            "forward_mps": float(velocity[0]),
            "energy_j": energy,
            "depth_error_m": depth_error,
            "upright": upright,
            "unstable": unstable,
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from ee_jumper.viewer import PassiveViewer

                self.viewer = PassiveViewer(self.model, self.data)
                self.viewer.cam.distance = 1.25
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -30
            self.viewer.cam.lookat[:] = self.data.xpos[self.base]
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self.data.xpos[self.base]
            camera.distance, camera.azimuth, camera.elevation = 1.15, 135, -35
            self.renderer.update_scene(self.data, camera)
            return self.renderer.render()
        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
