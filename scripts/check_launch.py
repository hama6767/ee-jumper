"""Exercise ros2 launch, the integrated GUI, keyboard commands, idle, and clean shutdown.

Run under xvfb-run on headless Linux; python-xlib is a test-only dependency.
Source the colcon install/setup.bash first.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np


def send_key(connection, window, key):
    from Xlib import XK, X
    from Xlib.protocol import event

    code = connection.keysym_to_keycode(XK.string_to_keysym(key))
    window.set_input_focus(X.RevertToParent, X.CurrentTime)
    for kind in (event.KeyPress, event.KeyRelease):
        window.send_event(
            kind(
                time=X.CurrentTime,
                root=connection.screen().root,
                window=window,
                same_screen=1,
                child=X.NONE,
                root_x=1,
                root_y=1,
                event_x=1,
                event_y=1,
                state=0,
                detail=code,
            )
        )
    connection.sync()


def find_control_window(connection):
    def walk(window):
        title = window.get_wm_name() or ""
        if title == "Jumper":
            return window
        for child in window.query_tree().children:
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(connection.screen().root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/launch-check.json")
    parser.add_argument("--library", help="Override the installed library")
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(170 + os.getpid() % 30)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    import rclpy
    from geometry_msgs.msg import PointStamped, Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from Xlib import display

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    rclpy.init()
    node = rclpy.create_node("ee_jumper_launch_check")
    poses, commands, states, centers = [], [], [], []
    prefix = "/ee_jumper_launch_check"
    node.create_subscription(
        Odometry,
        prefix + "/odom",
        lambda msg: poses.append(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        ),
        10,
    )
    node.create_subscription(
        Twist,
        prefix + "/cmd_vel",
        lambda msg: commands.append([msg.linear.x, msg.linear.y, msg.linear.z]),
        10,
    )
    node.create_subscription(String, prefix + "/stroke", lambda msg: states.append(msg.data), 10)
    node.create_subscription(
        PointStamped,
        prefix + "/com_position",
        lambda msg: centers.append([msg.point.x, msg.point.y, msg.point.z]),
        10,
    )
    connection = display.Display()
    process = None

    def spin(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.03)

    try:
        with log_path.open("w") as log:
            argv = [
                "ros2",
                "launch",
                "ee_jumper",
                "swim.launch.py",
                "teleop:=gui",
                "viewer:=true",
                f"topic_prefix:={prefix}",
                "speed:=0.08",
            ]
            if args.library:
                argv.append(f"library:={Path(args.library).resolve()}")
            process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + 20
            while (not poses or not commands or not centers) and time.monotonic() < deadline:
                spin(0.1)
            assert poses and commands and centers, (
                f"Launch did not publish both topics; see {log_path}"
            )
            spin(1)
            idle_drift = float(np.max(np.linalg.norm(np.array(centers) - centers[0], axis=1)))
            assert idle_drift < 1e-8
            window = find_control_window(connection)
            assert window is not None, "Command window not found"
            initial = np.array(centers[-1])
            send_key(connection, window, "w")
            spin(4)
            assert any(c[0] == 0.08 for c in commands), "GUI did not publish W command"
            assert "+x" in states
            distance = float(np.array(centers[-1])[0] - initial[0])
            assert distance > 0.02, "Simulator did not move after GUI command"
            send_key(connection, window, "space")
            spin(1)
            assert commands[-1] == [0, 0, 0] and states[-1] == "idle"
            send_key(connection, window, "q")
            code = process.wait(timeout=8)
            assert code == 0, f"Launch exited with code {code}"
        content = log_path.read_text()
        assert (
            "Traceback" not in content
            and "SIGKILL" not in content
            and "process has died" not in content
        ), content
        result = {
            "passed": True,
            "real_viewer": True,
            "real_command_gui": True,
            "ros_domain_id": os.environ["ROS_DOMAIN_ID"],
            "idle_drift_m": idle_drift,
            "x_progress_after_4_wall_seconds_m": distance,
            "odometry_messages": len(poses),
            "com_messages": len(centers),
            "command_messages": len(commands),
            "space_stops_commands": True,
            "q_shuts_down_launch": True,
            "gui": "PySide6 with embedded MuJoCo",
            "launch_exit_code": code,
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=10)
        connection.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
