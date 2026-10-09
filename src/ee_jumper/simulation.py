"""One simulation loop shared by the desktop demo and ROS adapter."""

import mujoco
import numpy as np

from ee_jumper.control import CycleController
from ee_jumper.model import Water, make_model, step_water


class Swimmer:
    dt = 0.01

    def __init__(self, library, water=None):
        self.library = library
        self.water = water or Water()
        self.model = make_model("free", self.water)
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body("base_link").id
        self.reset()

    def reset(self, roll=0.0):
        """Reset at rest; optional roll in radians demonstrates passive recovery."""
        self.controller = CycleController(self.model, self.library)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.controller.stroke.qadr] = self.controller.stroke.home
        self.data.ctrl[:] = self.controller.stroke.home
        self.data.qpos[3:7] = [np.cos(roll / 2), np.sin(roll / 2), 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.origin = self.data.subtree_com[self.base].copy()
        self.last_com = self.origin.copy()
        self.velocity = np.zeros(3)
        self.energy = self.power = 0.0
        self.output = None

    def tick(self, now):
        self.output = self.controller.step(self.dt, self.velocity, now)
        self.data.ctrl[:] = self.output.joint_positions
        expected = self.data.time + self.dt
        work = step_water(
            self.model, self.data, round(self.dt / self.model.opt.timestep), measure_work=True
        )
        if (
            self.data.time < expected - 1e-8
            or not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise RuntimeError("MuJoCo instability: simulation halted")
        self.power = work / self.dt
        self.energy += work
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.base].reshape(3, 3)
        center = self.data.subtree_com[self.base]
        self.velocity = rotation.T @ ((center - self.last_com) / self.dt)
        self.last_com = center.copy()

    @property
    def displacement(self):
        return self.data.subtree_com[self.base] - self.origin

    @property
    def attitude(self):
        """Roll, pitch, yaw in degrees (intrinsic XYZ / extrinsic ZYX)."""
        r = self.data.xmat[self.base].reshape(3, 3)
        return np.degrees(
            [
                np.arctan2(r[2, 1], r[2, 2]),
                np.arcsin(np.clip(-r[2, 0], -1, 1)),
                np.arctan2(r[1, 0], r[0, 0]),
            ]
        )

    @property
    def status(self):
        if self.output is None:
            return "idle"
        return self.output.active_stroke or ("timeout" if self.output.command_stale else "idle")
