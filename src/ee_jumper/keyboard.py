"""ROS 2 terminal teleop: w/s X, a/d Y, r/f Z, space stop, q quit."""

import argparse
import math
import sys


class KeyboardCommand:
    KEYS = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "r": (2, 1), "f": (2, -1)}

    def __init__(self, speed=0.08):
        if not math.isfinite(speed) or not 0 < speed <= 0.1:
            raise ValueError("speed must be in (0, 0.1] m/s")
        self.speed = speed
        self.velocity = [0.0, 0.0, 0.0]

    def press(self, key):
        key = key.lower()
        if key in (" ", "space", "escape", "q"):
            self.velocity[:] = [0.0, 0.0, 0.0]
        elif key in self.KEYS:
            axis, sign = self.KEYS[key]
            self.velocity[axis] = sign * self.speed
        return key not in ("q", "escape")


def main():
    import select
    import termios
    import tty

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/ee_jumper/cmd_vel")
    parser.add_argument("--speed", type=float, default=0.08)
    args = parser.parse_args()
    command = KeyboardCommand(args.speed)
    if not sys.stdin.isatty():
        parser.error("Use an interactive terminal; the graphical demo is ee-jumper-gui")
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = rclpy.create_node("ee_jumper_keyboard")
    publisher = node.create_publisher(Twist, args.topic, 10)
    settings = termios.tcgetattr(sys.stdin)
    try:
        print("W/S: X, A/D: Y, R/F: Z. Commands latch. Space: coast, Q: quit.")
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if not command.press(sys.stdin.read(1)):
                    break
            message = Twist()
            message.linear.x, message.linear.y, message.linear.z = command.velocity
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            publisher.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
