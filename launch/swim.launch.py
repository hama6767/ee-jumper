"""Start the integrated PySide6 demo, or the headless / terminal ROS adapter."""

import json
import os
import shutil
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from launch import LaunchDescription


def start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    python = value("python")
    if not Path(python).is_file():
        raise RuntimeError(f"Python not found: {python}; set python:=/path/to/venv/bin/python")
    viewer = value("viewer").lower()
    if viewer not in ("true", "false"):
        raise ValueError("viewer must be true or false")
    teleop = value("teleop")
    if teleop not in ("gui", "terminal", "none"):
        raise ValueError("teleop must be gui, terminal, or none")
    prefix = value("topic_prefix").rstrip("/")
    graphical = teleop == "gui"
    sim = ExecuteProcess(
        cmd=[
            python,
            "-m",
            "ee_jumper.dashboard" if graphical else "ee_jumper.ros_bridge",
            "--library",
            value("library"),
            "--topic-prefix",
            prefix,
            "--duration",
            value("duration"),
            *(["--ros", "--speed", value("speed")] if graphical else []),
            *(["--no-viewer"] if graphical and viewer == "false" else []),
            *(["--viewer"] if not graphical and viewer == "true" else []),
        ],
        name="swimmer",
        output="screen",
        additional_env={"PYTHONUNBUFFERED": "1"},
    )
    actions = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=sim, on_exit=[EmitEvent(event=Shutdown(reason="Simulator exited"))]
            )
        ),
        sim,
    ]
    if teleop == "terminal":
        command = [
            python,
            "-m",
            "ee_jumper.keyboard",
            "--topic",
            prefix + "/cmd_vel",
            "--speed",
            value("speed"),
        ]
        terminal = shutil.which("xterm")
        if not terminal:
            raise RuntimeError("teleop:=terminal requires xterm; use teleop:=gui or none")
        command = [terminal, "-T", "EE Jumper keyboard", "-e", *command]
        keys = ExecuteProcess(cmd=command, name="xyz_commands", output="screen")
        actions += [
            RegisterEventHandler(
                OnProcessExit(
                    target_action=keys,
                    on_exit=[EmitEvent(event=Shutdown(reason="Command window exited"))],
                )
            ),
            keys,
        ]
    return actions


def generate_launch_description():
    share = Path(get_package_share_directory("ee_jumper"))
    configured = json.loads((share / "runtime.json").read_text())["python"]
    default_python = os.environ.get(
        "EE_JUMPER_PYTHON", configured if Path(configured).is_file() else sys.executable
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("viewer", default_value="true", description="Show MuJoCo viewer"),
            DeclareLaunchArgument(
                "teleop", default_value="gui", description="gui / terminal / none"
            ),
            DeclareLaunchArgument(
                "library", default_value=str(share / "examples/strokes-xyz.json")
            ),
            DeclareLaunchArgument(
                "python", default_value=default_python, description="Python with MuJoCo and rclpy"
            ),
            DeclareLaunchArgument(
                "speed", default_value="0.08", description="Target speed in m/s (0, 0.1]"
            ),
            DeclareLaunchArgument("topic_prefix", default_value="/ee_jumper"),
            DeclareLaunchArgument(
                "duration", default_value="0", description="Wall seconds; 0 runs until closed"
            ),
            OpaqueFunction(function=start),
        ]
    )
