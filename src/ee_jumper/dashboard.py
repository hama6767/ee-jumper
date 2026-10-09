"""PySide6 demo: one simulation, an embedded MuJoCo view, controls and telemetry."""

import argparse
import csv
import signal
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ee_jumper.keyboard import KeyboardCommand
from ee_jumper.learn_cycles import load_library
from ee_jumper.simulation import Swimmer

COLORS = ("#55dec7", "#ffb85e", "#8fa7ff")
STYLE = """
QWidget { background: #101c2a; color: #e3edf6; font: 12px sans-serif; }
QMainWindow { background: #101c2a; }
QLabel#title { font-size: 25px; font-weight: bold; }
QLabel#metric { font: 18px monospace; color: #55dec7; padding: 7px; background: #17283a; }
QLabel#muted { color: #99afc4; }
QPushButton { background: #22384c; border: 1px solid #37536c; border-radius: 5px; padding: 10px; }
QPushButton:hover { background: #30556a; }
QPushButton#stop { background: #753d42; border-color: #d87c7c; font-weight: bold; }
QDoubleSpinBox, QTableWidget { background: #17283a; border: 1px solid #2b4358; padding: 4px; }
QHeaderView::section { background: #22384c; color: #e3edf6; border: none; padding: 6px; }
QTabBar::tab { background: #17283a; padding: 8px 18px; }
QTabBar::tab:selected { background: #30556a; }
QProgressBar { background: #17283a; border: none; text-align: center; height: 16px; }
QProgressBar::chunk { background: #348b83; }
"""


class Plot(QWidget):
    """Small rolling plot; Qt painting keeps the GUI dependency to Essentials."""

    def __init__(self, title, names):
        super().__init__()
        self.title, self.names = title, names
        self.samples = np.empty((0, len(names) + 1))
        self.setMinimumSize(280, 145)

    def set_samples(self, samples):
        self.samples = np.asarray(samples)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#142336"))
        p.setPen(QColor("#e3edf6"))
        p.drawText(12, 20, self.title)
        for i, name in enumerate(self.names):
            p.setPen(QColor(COLORS[i % 3]))
            p.drawText(12 + i * 72, 38, name)
        area = QRectF(57, 49, self.width() - 73, self.height() - 74)
        if len(self.samples) < 2:
            return
        t, y = self.samples[:, 0], self.samples[:, 1:]
        lo, hi = min(0, float(y.min())), max(0, float(y.max()))
        span = max(hi - lo, 0.01)
        lo, hi = lo - span * 0.12, hi + span * 0.12
        left, right = max(0, t[-1] - 30), max(1, t[-1])
        for fraction in (0, 0.5, 1):
            yp = area.bottom() - fraction * area.height()
            p.setPen(QPen(QColor("#2b4054"), 1))
            p.drawLine(QPointF(area.left(), yp), QPointF(area.right(), yp))
            p.setPen(QColor("#99afc4"))
            p.drawText(
                QRectF(2, yp - 8, 50, 18),
                Qt.AlignmentFlag.AlignRight,
                f"{lo + fraction * (hi - lo):.2f}",
            )
        p.drawText(int(area.left()), self.height() - 7, f"{left:.0f} s")
        p.drawText(int(area.right() - 45), self.height() - 7, f"{right:.1f} s")
        p.setClipRect(area.adjusted(-1, -1, 1, 1))
        for column in range(y.shape[1]):
            path = QPainterPath()
            for i in range(len(t)):
                point = QPointF(
                    area.left() + (t[i] - left) / (right - left) * area.width(),
                    area.bottom() - (y[i, column] - lo) / (hi - lo) * area.height(),
                )
                path.moveTo(point) if i == 0 else path.lineTo(point)
            p.setPen(QPen(QColor(COLORS[column % 3]), 1.8))
            p.drawPath(path)


class Viewport(QLabel):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        sim.model.vis.global_.offwidth = 800
        sim.model.vis.global_.offheight = 450
        sim.model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(sim.model, height=450, width=800)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 0.95, 125, -27
        self.setMinimumSize(480, 270)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.last_mouse = None

    def draw(self):
        sim = self.sim
        self.camera.lookat[:] = sim.data.subtree_com[sim.base]
        self.renderer.update_scene(sim.data, self.camera)
        # Shadow maps are expensive on software OpenGL (e.g. laptop/Xvfb demos).
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        # Display the combined center of mass and center of buoyancy.
        center = sim.data.subtree_com[sim.base]
        up = sim.data.xmat[sim.base].reshape(3, 3)[:, 2]
        for point, color in (
            (center, [1, 0.45, 0.35, 1]),
            (center + sim.water.buoyancy_height_m * up, [0.2, 1, 0.85, 1]),
        ):
            scene = self.renderer.scene
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.004),
                point,
                np.eye(3).ravel(),
                np.array(color),
            )
            scene.ngeom += 1
        rgb = self.renderer.render()
        frame = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.setPixmap(
            QPixmap.fromImage(frame).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def mousePressEvent(self, event):
        self.last_mouse = event.position()

    def mouseMoveEvent(self, event):
        if self.last_mouse is not None:
            delta = event.position() - self.last_mouse
            self.camera.azimuth -= delta.x() * 0.4
            self.camera.elevation = np.clip(self.camera.elevation - delta.y() * 0.3, -89, 89)
            self.last_mouse = event.position()

    def mouseReleaseEvent(self, event):
        self.last_mouse = None

    def wheelEvent(self, event):
        self.camera.distance = np.clip(
            self.camera.distance * np.exp(-event.angleDelta().y() * 0.001), 0.25, 5
        )


class Dashboard(QMainWindow):
    def __init__(self, sim, *, ros=False, prefix="/ee_jumper", speed=0.08, viewer=True, duration=0):
        super().__init__()
        self.sim, self.command = sim, KeyboardCommand(speed)
        self.ros, self.node, self.publisher = ros, None, None
        self.duration, self.error = duration, None
        self.started = self.previous = self.origin_time = time.monotonic()
        self.accumulator = self.last_publish = self.last_sample = 0.0
        self.history = deque(maxlen=300)
        self.closed = False
        self.stop_requested = False
        self.setWindowTitle("EE Jumper · XYZ control · Swim lab")
        self.resize(1260, 900)
        self.setStyleSheet(STYLE)
        if ros:
            import rclpy
            from geometry_msgs.msg import Twist
            from rclpy.signals import SignalHandlerOptions

            from ee_jumper.ros_bridge import create_bridge

            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            self.node = create_bridge(sim, prefix)
            self.publisher = self.node.create_publisher(Twist, prefix.rstrip("/") + "/cmd_vel", 10)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        heading = QHBoxLayout()
        title = QLabel("EE JUMPER   /   SWIM LAB")
        title.setObjectName("title")
        heading.addWidget(title)
        self.state_label = QLabel()
        heading.addWidget(self.state_label, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(heading)
        subtitle = QLabel("Passive attitude stability  ·  Neutral buoyancy  ·  Six learned strokes")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        main = QHBoxLayout()
        self.view = Viewport(sim) if viewer else None
        main.addWidget(self.view if viewer else QLabel("3D rendering disabled"), 3)
        panel = QVBoxLayout()
        self.local = QCheckBox("Local keyboard / buttons")
        self.local.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.local.setChecked(True)
        self.local.setEnabled(ros)
        self.local.toggled.connect(self.change_authority)
        panel.addWidget(self.local)
        hint = QLabel(
            "W/S = X   A/D = Y   R/F = Z\nCommands latch; Space stops stroking.\nDrag: orbit camera  ·  Scroll: zoom"
        )
        hint.setObjectName("muted")
        panel.addWidget(hint)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.01, 0.10)
        self.speed.setSingleStep(0.01)
        self.speed.setValue(speed)
        self.speed.setSuffix(" m/s target")
        self.speed.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.speed.valueChanged.connect(self.change_speed)
        panel.addWidget(self.speed)
        keys = QGridLayout()
        for row, (a, b, axis) in enumerate((("w", "s", "X"), ("a", "d", "Y"), ("r", "f", "Z"))):
            for col, (key, sign) in enumerate(((a, "+"), (b, "−"))):
                button = QPushButton(f"{sign}{axis}   [{key.upper()}]")
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                button.clicked.connect(lambda checked=False, k=key: self.press(k))
                keys.addWidget(button, row, col)
        panel.addLayout(keys)
        stop = QPushButton("STOP STROKING   [Space]")
        stop.setObjectName("stop")
        stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        stop.clicked.connect(lambda: self.press("space"))
        panel.addWidget(stop)
        self.command_label = QLabel()
        panel.addWidget(self.command_label)
        self.phase = QProgressBar()
        self.phase.setRange(0, 100)
        panel.addWidget(self.phase)
        for text, fn in (
            ("Reset at rest", self.reset),
            ("Tilt 25° · passive recovery", lambda: self.reset(25)),
            ("Export last 30 s as CSV", self.export_csv),
        ):
            button = QPushButton(text)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda checked=False, action=fn: action())
            panel.addWidget(button)
        main.addLayout(panel, 1)
        layout.addLayout(main, 3)
        markers = QLabel(
            "COM: coral  ·  Buoyancy center: mint (+10 mm along body Z)  |  Q: close demo"
        )
        markers.setObjectName("muted")
        layout.addWidget(markers)
        cards = QHBoxLayout()
        self.cards = []
        for _ in range(4):
            label = QLabel()
            label.setObjectName("metric")
            self.cards.append(label)
            cards.addWidget(label)
        layout.addLayout(cards)
        tabs = QTabWidget()
        layout.addWidget(tabs, 2)
        plots = QWidget()
        grid = QGridLayout(plots)
        self.plots = [
            Plot("COM velocity / body frame · m/s", ("X", "Y", "Z")),
            Plot("Attitude · degrees", ("roll", "pitch", "yaw")),
            Plot("COM displacement / world frame · m", ("X", "Y", "Z")),
            Plot("Actuator mechanical power · W", ("Σ |τ ω|",)),
        ]
        for i, plot in enumerate(self.plots):
            grid.addWidget(plot, i // 2, i % 2)
        tabs.addTab(plots, "Motion & energy")
        self.joints = QTableWidget(sim.model.nu, 4)
        self.joints.setHorizontalHeaderLabels(["Joint", "Actual °", "Target °", "Torque N·m"])
        self.joints.horizontalHeader().setStretchLastSection(True)
        self.joints.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(sim.controller.stroke.names):
            for col in range(4):
                self.joints.setItem(row, col, QTableWidgetItem(name if col == 0 else "0"))
        tabs.addTab(self.joints, "22 actuators")
        self.statusBar().showMessage(
            "Ready · targets use body axes; displacement uses world axes. Stop allows coasting."
        )
        # Window shortcuts also work while the speed editor or joint table has focus.
        self.shortcuts = []
        for sequence, key in [(k.upper(), k) for k in KeyboardCommand.KEYS] + [
            ("Space", "space"),
            ("Q", "q"),
            ("Escape", "escape"),
        ]:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(lambda k=key: self.press(k))
            self.shortcuts.append(shortcut)
        self.physics = QTimer(self)
        self.physics.setInterval(10)
        self.physics.timeout.connect(self.advance)
        self.display = QTimer(self)
        self.display.setInterval(50)
        self.display.timeout.connect(self.refresh)
        self.physics.start()
        self.display.start()
        self.refresh()

    def change_authority(self, local):
        self.command.press("space")
        # Send one zero when releasing local control, then stop publishing.
        self.send_command()

    def change_speed(self, speed):
        self.command.speed = speed
        self.command.velocity[:] = [float(np.sign(v) * speed) for v in self.command.velocity]

    def press(self, key):
        if key in KeyboardCommand.KEYS and not self.local.isChecked():
            self.local.setChecked(True)
        keep = self.command.press(key)
        if self.local.isChecked() or key in ("space", " ", "q", "escape"):
            self.send_command()
        if not keep:
            self.close()

    def keyPressEvent(self, event):
        key = {Qt.Key.Key_Space: "space", Qt.Key.Key_Escape: "escape"}.get(
            event.key(), event.text().lower()
        )
        if key in KeyboardCommand.KEYS or key in ("space", "q", "escape"):
            self.press(key)
            event.accept()
        else:
            super().keyPressEvent(event)

    def send_command(self):
        if self.ros:
            from geometry_msgs.msg import Twist

            message = Twist()
            message.linear.x, message.linear.y, message.linear.z = self.command.velocity
            self.publisher.publish(message)
        else:
            self.sim.controller.set_command(self.command.velocity, time.monotonic())

    def reset(self, tilt=0):
        self.command.press("space")
        self.send_command()
        self.sim.reset(np.deg2rad(tilt))
        self.history.clear()
        self.last_sample = self.accumulator = 0
        self.origin_time = self.previous = time.monotonic()
        self.refresh()

    def advance(self):
        if self.closed:
            return
        if self.stop_requested:
            self.close()
            return
        try:
            now = time.monotonic()
            if self.duration > 0 and now - self.started >= self.duration:
                self.close()
                return
            if self.ros:
                import rclpy

                if not rclpy.ok():
                    self.close()
                    return
                # Drain a bounded amount of input without starving the Qt event loop.
                for _ in range(4):
                    rclpy.spin_once(self.node, timeout_sec=0)
            if self.local.isChecked() and now - self.last_publish >= 0.1:
                self.send_command()
                self.last_publish = now
            self.accumulator += min(now - self.previous, 0.10)
            self.previous = now
            while self.accumulator >= self.sim.dt:
                self.sim.tick(now)
                if self.node is not None:
                    self.node.publish()
                self.accumulator -= self.sim.dt
        except Exception as exc:
            self.error = str(exc)
            print(f"Demo stopped: {exc}", file=sys.stderr, flush=True)
            self.close()

    def refresh(self):
        if self.closed:
            return
        s = self.sim
        if self.view is not None:
            self.view.draw()
        elapsed = time.monotonic() - self.origin_time
        self.state_label.setText(
            f"{'ROS 2' if self.ros else 'STANDALONE'}  ·  {s.data.time:.1f} s  ·  {s.data.time / max(elapsed, 0.01):.2f}× realtime"
        )
        self.command_label.setText(
            "Body target m/s:  " + "  ".join(f"{v:+.2f}" for v in s.controller.command)
        )
        self.phase.setValue(round(s.output.phase * 100) if s.output else 0)
        self.phase.setFormat(f"Stroke {s.status}  ·  %p%")
        values = [
            f"NET TRAVEL\n{np.linalg.norm(s.displacement) * 100:.1f} cm",
            f"BODY SPEED\n{np.linalg.norm(s.velocity) * 100:.1f} cm/s",
            f"ROLL / PITCH\n{s.attitude[0]:+.1f}° / {s.attitude[1]:+.1f}°",
            f"MECH. WORK\n{s.energy:.2f} J",
        ]
        for card, value in zip(self.cards, values):
            card.setText(value)
        if s.data.time - self.last_sample >= 0.1 or not self.history:
            self.last_sample = s.data.time
            self.history.append(
                [s.data.time, *s.velocity, *s.attitude, *s.displacement, s.power, s.energy]
            )
            a = np.array(self.history)
            for plot, cols in zip(self.plots, ([0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9], [0, 10])):
                plot.set_samples(a[:, cols])
        for i, (actual, target, torque) in enumerate(
            zip(
                np.degrees(s.data.qpos[s.controller.stroke.qadr]),
                np.degrees(s.data.ctrl),
                s.data.actuator_force,
            )
        ):
            for col, value in enumerate((actual, target, torque), 1):
                self.joints.item(i, col).setText(f"{value:+.2f}")
            self.joints.item(i, 3).setForeground(
                QColor("#ffb85e" if abs(torque) > 1.8 else "#55dec7")
            )

    def export_csv(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export telemetry", "ee-jumper.csv", "CSV (*.csv)"
        )
        if filename:
            with Path(filename).open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "time_s",
                        "body_vx_mps",
                        "body_vy_mps",
                        "body_vz_mps",
                        "roll_deg",
                        "pitch_deg",
                        "yaw_deg",
                        "world_dx_m",
                        "world_dy_m",
                        "world_dz_m",
                        "mechanical_power_w",
                        "mechanical_work_j",
                    ]
                )
                writer.writerows(self.history)

    def closeEvent(self, event):
        if not self.closed:
            self.closed = True
            self.physics.stop()
            self.display.stop()
            self.command.press("space")
            if self.ros:
                import rclpy

                if rclpy.ok():
                    self.send_command()
                self.node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            if self.view is not None:
                self.view.renderer.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--ros", action="store_true")
    parser.add_argument("--topic-prefix", default="/ee_jumper")
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = Dashboard(
        Swimmer(load_library(args.library)),
        ros=args.ros,
        prefix=args.topic_prefix,
        speed=args.speed,
        viewer=not args.no_viewer,
        duration=args.duration,
    )
    # Qt callbacks must not swallow KeyboardInterrupt during launch shutdown.
    previous = {
        sig: signal.signal(sig, lambda *_: setattr(window, "stop_requested", True))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    window.show()
    try:
        app.exec()
    finally:
        window.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # Destroy widgets while QApplication is still alive.
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    if window.error:
        raise SystemExit(window.error)


if __name__ == "__main__":
    main()
