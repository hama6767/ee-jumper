"""A 13-parameter symmetric oscillator policy, fitted using CEM, not feedback RL."""

import numpy as np

N_PARAMETERS = 13


def action_at(time, parameters, joint_names):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (N_PARAMETERS,) or not np.isfinite(parameters).all():
        raise ValueError(f"Expected {N_PARAMETERS} finite gait parameters")
    p = np.clip(parameters, -1.0, 1.0)
    frequency = 0.8 + 0.5 * p[0]  # 0.3–1.3 Hz
    result = np.zeros(len(joint_names), dtype=np.float32)
    index = {name: i for i, name in enumerate(joint_names)}
    for group, pair in enumerate((("LF", "RF"), ("LM", "RM"), ("LR", "RR"))):
        sweep, fold, lag, offset = p[1 + 4 * group : 5 + 4 * group]
        phase = 2.0 * np.pi * frequency * time + np.pi * offset
        sweep_angle = sweep * np.sin(phase)
        fold_angle = fold * np.sin(phase + np.pi * lag)
        for side, sign in zip(pair, (1, -1)):
            result[index[f"{side}_J0_joint"]] = sign * sweep_angle
            if group == 0:
                # Front J1 axes point along y on both arms. J2/J3 are mirrored x axes.
                result[index[f"{side}_J2_joint"]] = sign * fold_angle
                result[index[f"{side}_J3_joint"]] = -sign * 0.5 * fold_angle
            else:
                result[index[f"{side}_J1_joint"]] = sign * fold_angle
                result[index[f"{side}_J2_joint"]] = -sign * 0.7 * fold_angle
    return result


def rollout(env, parameters=None, seed=0, callback=None):
    env.reset(seed=seed)
    total_reward = total_energy = 0.0
    for step in range(env.max_steps):
        action = (
            np.zeros(env.model.nu, dtype=np.float32)
            if parameters is None
            else action_at(step * env.dt, parameters, env.joint_names)
        )
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_energy += info["energy_j"]
        if callback is not None:
            callback(step, env, info)
        if terminated or truncated:
            break
    return {
        **info,
        "return": total_reward,
        "energy_j": total_energy,
        "seconds": (step + 1) * env.dt,
        "terminated": terminated,
        "mean_forward_mps": info["x_m"] / ((step + 1) * env.dt),
    }
