"""Thin ROS 2 adapter around the shared simulation; ROS is an optional dependency."""

import argparse
import time

import mujoco
import numpy as np

from ee_jumper.learn_cycles import load_library
from ee_jumper.simulation import Swimmer


def create_bridge(sim, prefix="/ee_jumper"):
    from geometry_msgs.msg import PointStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    model, data, base, library = sim.model, sim.data, sim.base, sim.library
    prefix = prefix.rstrip("/")

    class Bridge(Node):
        def __init__(self):
            super().__init__("ee_jumper_swimmer")
            self.create_subscription(Twist, prefix + "/cmd_vel", self.command, 10)
            self.joint_pub = self.create_publisher(JointState, prefix + "/joint_states", 10)
            self.target_pub = self.create_publisher(JointTrajectory, prefix + "/joint_targets", 10)
            self.odom_pub = self.create_publisher(Odometry, prefix + "/odom", 10)
            self.com_pub = self.create_publisher(PointStamped, prefix + "/com_position", 1)
            self.status_pub = self.create_publisher(String, prefix + "/stroke", 10)
            self.get_logger().info(
                f"Listening on {prefix}/cmd_vel (body XYZ m/s); period {library['period_s']} s"
            )

        def command(self, message):
            try:
                sim.controller.set_command(
                    [message.linear.x, message.linear.y, message.linear.z], time.monotonic()
                )
            except ValueError as exc:
                self.get_logger().warning(str(exc))

        def tick(self):
            sim.tick(time.monotonic())
            self.publish()

        def publish(self):
            controller, output = sim.controller, sim.output
            stamp = self.get_clock().now().to_msg()
            state = JointState()
            state.header.stamp, state.name = stamp, controller.stroke.names
            state.position, state.velocity = (
                data.qpos[controller.stroke.qadr].tolist(),
                data.qvel[controller.stroke.vadr].tolist(),
            )
            state.effort = data.actuator_force.tolist()
            self.joint_pub.publish(state)
            target = JointTrajectory()
            target.header.stamp, target.joint_names = stamp, controller.stroke.names
            point = JointTrajectoryPoint()
            point.positions = output.joint_positions.tolist()
            point.time_from_start.nanosec = 10_000_000
            target.points = [point]
            self.target_pub.publish(target)
            odom = Odometry()
            odom.header.stamp, odom.header.frame_id, odom.child_frame_id = (
                stamp,
                "world",
                "base_link",
            )
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(
                float, data.xpos[base]
            )
            q = data.qpos[3:7]
            (
                odom.pose.pose.orientation.w,
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
            ) = map(float, q)
            body_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, body_velocity, 1)
            odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = (
                map(float, body_velocity[:3])
            )
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(
                float, body_velocity[3:]
            )
            self.odom_pub.publish(odom)
            center = PointStamped()
            center.header.stamp, center.header.frame_id = stamp, "world"
            center.point.x, center.point.y, center.point.z = map(float, data.subtree_com[base])
            self.com_pub.publish(center)
            status = String()
            status.data = output.active_stroke or ("timeout" if output.command_stale else "idle")
            self.status_pub.publish(status)

    return Bridge()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--viewer", action="store_true", help="Legacy native MuJoCo viewer")
    parser.add_argument("--topic-prefix", default="/ee_jumper")
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    import rclpy
    from rclpy.executors import ExternalShutdownException

    sim = Swimmer(load_library(args.library))
    viewer = None
    if args.viewer:
        from ee_jumper.viewer import PassiveViewer

        viewer = PassiveViewer(sim.model, sim.data)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, 135, -30
    rclpy.init()
    node = create_bridge(sim, args.topic_prefix)

    def tick():
        node.tick()
        if viewer is not None:
            viewer.cam.lookat[:] = sim.data.xpos[sim.base]
            viewer.sync()

    node.create_timer(sim.dt, tick)
    start = time.monotonic()
    try:
        while rclpy.ok() and (args.duration <= 0 or time.monotonic() - start < args.duration):
            rclpy.spin_once(node, timeout_sec=0.1)
            if viewer is not None and not viewer.is_running():
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
