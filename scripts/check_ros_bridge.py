"""Integration check with real ROS 2 messages on an isolated localhost domain."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", default="runs/ros-check.json")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(170 + os.getpid() % 30)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory

    from ee_jumper.cycles import StrokeSpace
    from ee_jumper.model import make_model

    home = StrokeSpace(make_model("free")).home
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    rclpy.init()
    node = rclpy.create_node("ee_jumper_integration_check")
    publisher = node.create_publisher(Twist, "/ee_jumper_smoke/cmd_vel", 10)
    targets, states, positions = [], [], []
    node.create_subscription(
        JointTrajectory,
        "/ee_jumper_smoke/joint_targets",
        lambda msg: targets.append(msg.points[0].positions),
        10,
    )
    node.create_subscription(
        String, "/ee_jumper_smoke/stroke", lambda msg: states.append(msg.data), 10
    )
    node.create_subscription(
        Odometry,
        "/ee_jumper_smoke/odom",
        lambda msg: positions.append(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        ),
        10,
    )
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ee_jumper.ros_bridge",
                    "--library",
                    args.library,
                    "--topic-prefix",
                    "/ee_jumper_smoke",
                    "--duration",
                    "16",
                    *(["--viewer"] if args.viewer else []),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 6
            while not positions and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            assert positions, f"No ROS odometry received; see {log_path}"
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            idle_drift = float(np.max(np.linalg.norm(np.array(positions) - positions[0], axis=1)))
            assert idle_drift < 1e-8, f"Robot drifts without commands: {idle_drift} m"
            start = time.monotonic()
            while time.monotonic() - start < 3:
                command = Twist()
                command.linear.x = 0.05
                publisher.publish(command)
                for _ in range(5):
                    rclpy.spin_once(node, timeout_sec=0.02)
            # No messages: test actual watchdog behavior, not a zero command.
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            assert "+x" in states, "No learned +X stroke selected"
            assert any(np.linalg.norm(np.array(t) - home) > 0.01 for t in targets), (
                "No joint motion"
            )
            assert states[-1] == "timeout", "Watchdog did not report timeout"
            np.testing.assert_allclose(targets[-1], home, atol=1e-8)
            assert np.isfinite(positions).all()
            result = {
                "ros_domain_id": os.environ["ROS_DOMAIN_ID"],
                "odometry_messages": len(positions),
                "target_messages": len(targets),
                "selected_strokes": sorted(set(states)),
                "watchdog_home_return": True,
                "real_viewer": args.viewer,
                "idle_seconds": 2,
                "idle_max_displacement_m": idle_drift,
                "first_position_m": positions[0],
                "last_position_m": positions[-1],
                "passed": True,
            }
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
