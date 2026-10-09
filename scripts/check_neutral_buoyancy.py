"""Verify upright rest and passive roll/pitch recovery, including viewer sync."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ee_jumper.cycles import DynamicBench
from ee_jumper.model import step_water


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", default="runs/neutral-buoyancy.json")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 10:
        parser.error("--seconds must be at least 10 for the recovery check")
    bench = DynamicBench(mode="free")
    m, d = bench.model, bench.data
    base = m.body("base_link").id
    viewer = None
    if args.viewer:
        from ee_jumper.viewer import PassiveViewer

        viewer = PassiveViewer(m, d)
    records = []
    try:
        for name, axis in (("upright", 0), ("roll_25deg", 1), ("pitch_25deg", 2)):
            bench.reset()
            if axis:
                d.qpos[3:7] = [np.cos(np.deg2rad(12.5)), 0, 0, 0]
                d.qpos[3 + axis] = np.sin(np.deg2rad(12.5))
            mujoco.mj_forward(m, d)
            initial = d.qpos.copy()
            com = d.subtree_com[base].copy()
            trace = []
            peak_error = 0.0
            for step in range(round(args.seconds / bench.dt)):
                if viewer is not None:
                    viewer.sync()
                else:
                    d.xfrc_applied[:] = 0
                step_water(m, d, bench.substeps)
                peak_error = max(peak_error, float(np.max(abs(d.qpos - initial))))
                if step % 100 == 99:
                    mujoco.mj_forward(m, d)
                    tilt = float(
                        np.degrees(np.arccos(np.clip(d.xmat[base].reshape(3, 3)[2, 2], -1, 1)))
                    )
                    trace.append({"seconds": round(d.time, 3), "tilt_deg": tilt})
            mujoco.mj_forward(m, d)
            record = {
                "case": name,
                "seconds": float(d.time),
                "final_tilt_deg": trace[-1]["tilt_deg"],
                "displacement_m": (d.subtree_com[base] - com).tolist(),
                "max_qpos_change": peak_error,
                "trace": trace,
            }
            records.append(record)
            if not axis:
                assert peak_error < 1e-8, record
            else:
                assert max(t["tilt_deg"] for t in trace[-5:]) < 1, record
            np.testing.assert_array_equal(d.ctrl, bench.stroke.home)
    finally:
        if viewer is not None:
            viewer.close()
    result = {
        "mujoco_version": mujoco.__version__,
        "real_viewer": args.viewer,
        "buoyancy_height_m": 0.01,
        "cases": records,
        "passed": True,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({r["case"]: r["final_tilt_deg"] for r in records}, indent=2))


if __name__ == "__main__":
    main()
