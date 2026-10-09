"""Pure Python XYZ command -> measured-stroke scheduling, independent of ROS.

Each full stroke is kept intact. Concurrent axes use whole-cycle time sharing;
we do NOT assume that adding two joint trajectories adds their fluid forces.
"""

from dataclasses import dataclass

import numpy as np

from ee_jumper.cycles import DIRECTIONS, StrokeSpace


def project_duty(values):
    """Euclidean projection onto weights >= 0, sum(weights) <= 1."""
    positive = np.maximum(values, 0)
    if positive.sum() <= 1:
        return positive
    sorted_values = np.sort(values)[::-1]
    offsets = (np.cumsum(sorted_values) - 1) / np.arange(1, len(values) + 1)
    active = np.flatnonzero(sorted_values > offsets)[-1]
    return np.maximum(values - offsets[active], 0)


@dataclass
class ControlOutput:
    joint_positions: np.ndarray
    active_stroke: str | None
    phase: float
    duties: np.ndarray
    desired_force_n: np.ndarray
    command_stale: bool


class CycleController:
    def __init__(
        self, model, library, command_timeout=0.5, max_speed=0.1, velocity_time_constant=2.0
    ):
        if command_timeout <= 0 or max_speed <= 0 or velocity_time_constant <= 0:
            raise ValueError("Control limits must be positive")
        self.use_velocity_allocation = library["format"] == "ee-jumper-strokes-v2"
        self.stroke = StrokeSpace(model, library["period_s"], library.get("max_excursion_rad", 0.9))
        self.names = list(DIRECTIONS)
        self.parameters = [np.array(library["strokes"][name]["parameters"]) for name in self.names]
        self.forces = np.array(
            [library["strokes"][name]["metrics"]["mean_force_n"] for name in self.names]
        ).T
        moments = np.array(
            [library["strokes"][name]["metrics"]["mean_torque_nm"] for name in self.names]
        ).T
        # Suppress parasitic moments during allocation, without claiming attitude control.
        self.allocation = np.vstack((self.forces, moments / 0.1))
        self.lipschitz = float(np.linalg.norm(self.allocation, 2) ** 2 + 1e-8)
        self.speed_matrix = None
        validation = [library["strokes"][name].get("free_validation") for name in self.names]
        if all(validation):
            velocities = [
                np.array(record["displacement_m"]) / record["elapsed_s"] for record in validation
            ]
            matrix = np.asarray(velocities).T
            if matrix.shape == (3, 6) and np.isfinite(matrix).all():
                self.speed_matrix = matrix
        self.feedforward_force = np.zeros(3)
        self.mass = float(model.body_mass.sum())
        self.command_timeout, self.max_speed = command_timeout, max_speed
        self.velocity_time_constant = velocity_time_constant
        self.command = np.zeros(3)
        self.command_time = -np.inf
        self.credit = np.zeros(7)  # six directions + idle
        self.phase, self.active = 0.0, None
        self.last_target = self.stroke.home.copy()
        self.stopping_from = None
        self.stopping_time = 0.0
        self.duties = np.zeros(6)
        self.filtered_velocity = np.zeros(3)
        self.integrated_error = np.zeros(3)

    def set_command(self, velocity, now):
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all() or not np.isfinite(now):
            raise ValueError("Command must be three finite body-frame velocities and a finite time")
        if np.linalg.norm(velocity - self.command) > 0.01:
            self.credit[:] = 0
        command = np.clip(velocity, -self.max_speed, self.max_speed)
        if self.speed_matrix is not None and not np.array_equal(command, self.command):
            # Calibrate the duty needed to sustain motion from actual swimming.
            # Tethered thrust alone overestimates useful propulsion and can
            # give powerful strokes excessive idle time. This only schedules
            # joint trajectories; no artificial force is applied to the robot.
            duty = self._solve_duty(self.speed_matrix, command)
            self.feedforward_force = self.forces @ duty
        self.command = command
        self.command_time = now

    def allocate(self, force):
        return self._solve_duty(self.allocation, np.r_[force, np.zeros(3)], self.lipschitz)

    @staticmethod
    def _solve_duty(matrix, desired, lipschitz=None):
        if lipschitz is None:
            lipschitz = float(np.linalg.norm(matrix, 2) ** 2 + 1e-8)
        duty = np.zeros(6)
        for _ in range(100):
            gradient = matrix.T @ (matrix @ duty - desired)
            duty = project_duty(duty - gradient / lipschitz)
        return duty

    def step(self, dt, measured_velocity, now):
        velocity = np.asarray(measured_velocity, dtype=float)
        if not np.isfinite(dt) or dt <= 0 or dt > self.stroke.period:
            raise ValueError("dt must be positive and no longer than one stroke")
        if velocity.shape != (3,) or not np.isfinite(velocity).all() or not np.isfinite(now):
            raise ValueError("Velocity feedback and time must be finite")
        stale = now - self.command_time > self.command_timeout
        stopped = stale or np.linalg.norm(self.command) < 1e-6
        # Reject intra-stroke velocity oscillations; they are not the desired
        # cycle-average speed. Bounded integral feedback reduces steady error.
        self.filtered_velocity += (
            dt / (self.stroke.period + dt) * (velocity - self.filtered_velocity)
        )
        error = self.command - self.filtered_velocity
        if stopped:
            self.integrated_error[:] = 0
        else:
            self.integrated_error = np.clip(self.integrated_error + error * dt, -0.06, 0.06)
        desired = (
            np.zeros(3)
            if stopped
            else self.feedforward_force
            + self.mass * (error + self.integrated_error / 4.0) / self.velocity_time_constant
        )
        if stopped:
            if self.stopping_from is None:
                self.stopping_from = self.last_target.copy()
                self.stopping_time = 0.0
            self.stopping_time += dt
            u = min(1.0, self.stopping_time / 0.3)
            blend = u**3 * (10 - 15 * u + 6 * u * u)
            target = (1 - blend) * self.stopping_from + blend * self.stroke.home
            self.active, self.phase = None, 0.0
            self.credit[:] = 0
            self.duties[:] = 0
        else:
            # Complete the home return before launching another full stroke.
            if self.stopping_from is not None and self.stopping_time < 0.3:
                self.stopping_time += dt
                u = min(1.0, self.stopping_time / 0.3)
                blend = u**3 * (10 - 15 * u + 6 * u * u)
                target = (1 - blend) * self.stopping_from + blend * self.stroke.home
                self.last_target = target.copy()
                return ControlOutput(target, None, 0.0, self.duties.copy(), desired, stale)
            self.stopping_from = None
            if self.phase < 1e-10:
                if self.use_velocity_allocation and self.speed_matrix is not None:
                    # V2 is trained and allocated in measured free-swim velocity
                    # space. Tether moments no longer suppress useful strokes.
                    target_velocity = self.command + 0.5 * (error + self.integrated_error / 4)
                    self.duties = self._solve_duty(self.speed_matrix, target_velocity)
                    desired = self.forces @ self.duties
                else:
                    self.duties = self.allocate(desired)
                weights = np.r_[self.duties, max(0, 1 - self.duties.sum())]
                self.credit += weights
                eligible = np.where(weights > 1e-8, self.credit, -np.inf)
                self.active = int(np.argmax(eligible))
                self.credit[self.active] -= 1
            if self.active is not None and self.active < 6:
                target = self.stroke.trajectory([self.phase], self.parameters[self.active])[0][0]
            else:
                target = self.stroke.home.copy()
            self.phase += dt / self.stroke.period
            if self.phase >= 1 - 1e-10:
                self.phase = 0.0
        self.last_target = target.copy()
        label = self.names[self.active] if self.active is not None and self.active < 6 else None
        return ControlOutput(target, label, self.phase, self.duties.copy(), desired, stale)
