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
    QAction,
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
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ee_jumper.keyboard import KeyboardCommand
from ee_jumper.learn_cycles import load_library
from ee_jumper.simulation import Swimmer

COLORS = ("#ce6562", "#82ae83", "#77a6d1")
STYLE = """
QWidget { background: #252729; color: #e1e1e1; font: 12px sans-serif; }
QToolBar { background: #303234; border: none; spacing: 6px; padding: 3px; }
QLabel#readout { font: 13px monospace; padding: 4px 0; }
QLabel#muted { color: #a8aaac; }
QPushButton { background: #393b3d; border: 1px solid #505254; border-radius: 2px; padding: 7px; }
QPushButton:hover { background: #484b4e; }
QPushButton:checked { background: #505860; }
QPushButton#stop { background: #713e3e; border-color: #965353; }
QDoubleSpinBox, QTableWidget { background: #202224; border: 1px solid #505254; padding: 4px; }
QHeaderView::section { background: #303234; color: #ddd; border: none; padding: 5px; }
QTabBar::tab { background: #303234; padding: 7px 16px; }
QTabBar::tab:selected { background: #45494c; }
QProgressBar { background: #202224; border: 1px solid #414447; text-align: center; height: 16px; }
QProgressBar::chunk { background: #51606b; }
QSplitter::handle { background: #3d4042; height: 4px; }
"""


class Plot(QWidget):
    """Small rolling plot; Qt painting keeps the GUI dependency to Essentials."""

    def __init__(self, title, names):
        super().__init__()
        self.title, self.names = title, names
        self.samples = np.empty((0, len(names) + 1))
        self.setMinimumSize(230, 150)

    def set_samples(self, samples):
        self.samples = np.asarray(samples)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#242628"))
        p.setPen(QColor("#dddddd"))
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
            p.setPen(QPen(QColor("#424548"), 1))
            p.drawLine(QPointF(area.left(), yp), QPointF(area.right(), yp))
            p.setPen(QColor("#a8aaac"))
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
        sim.model.vis.global_.offwidth = 960
        sim.model.vis.global_.offheight = 540
        sim.model.vis.quality.offsamples = 2
        self.renderer = mujoco.Renderer(sim.model, height=540, width=960)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 0.95, 125, -27
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.trail = deque(maxlen=240)
        self.trail_time = -1.0
        self.show_trail = True
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.last_mouse = None

    def draw(self):
        sim = self.sim
        self.camera.lookat[:] = sim.data.subtree_com[sim.base]
        self.renderer.update_scene(sim.data, self.camera)
        # Shadow maps are expensive on software OpenGL (e.g. laptop/Xvfb demos).
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        # A world-fixed COM trail makes translation visible with a following camera.
        if sim.data.time < self.trail_time:
            self.trail.clear()
            self.trail_time = -1.0
        if sim.data.time - self.trail_time >= 0.1:
            self.trail.append(sim.data.subtree_com[sim.base].copy())
            self.trail_time = sim.data.time
        if self.show_trail:
            scene = self.renderer.scene
            points = list(self.trail)
            for start, end in zip(points, points[1:]):
                if np.linalg.norm(end - start) < 1e-5:
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([0.87, 0.88, 0.90, 0.85]),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, start, end)
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
        self.setWindowTitle("Jumper")
        self.resize(1360, 850)
        self.setStyleSheet(STYLE)
        if ros:
            import rclpy
            from geometry_msgs.msg import Twist
            from rclpy.signals import SignalHandlerOptions

            from ee_jumper.ros_bridge import create_bridge

            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            self.node = create_bridge(sim, prefix)
            self.publisher = self.node.create_publisher(Twist, prefix.rstrip("/") + "/cmd_vel", 10)
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        reset_action = toolbar.addAction("Reset")
        reset_action.triggered.connect(lambda: self.reset())
        tilt_action = toolbar.addAction("Tilt 25°")
        tilt_action.triggered.connect(lambda: self.reset(25))
        toolbar.addSeparator()
        self.view = Viewport(sim) if viewer else None
        for label, azimuth, elevation in (
            ("Perspective", 125, -27),
            ("Side", 90, -5),
            ("Top", 90, -89),
        ):
            action = toolbar.addAction(label)
            action.triggered.connect(
                lambda checked=False, a=azimuth, e=elevation: self.set_view(a, e)
            )
        toolbar.addSeparator()
        self.presentation = QAction("Presentation [Tab]", self)
        self.presentation.setCheckable(True)
        self.presentation.toggled.connect(self.set_presentation)
        toolbar.addAction(self.presentation)
        full = toolbar.addAction("Full screen [F11]")
        full.triggered.connect(self.toggle_fullscreen)
        toolbar.addSeparator()
        toolbar.addAction("Save CSV").triggered.connect(self.export_csv)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 2)
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)
        stage = QWidget()
        main = QHBoxLayout(stage)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self.view if viewer else QLabel("3D view disabled"), 1)
        self.controls = QWidget()
        self.controls.setFixedWidth(224)
        panel = QVBoxLayout(self.controls)
        panel.setContentsMargins(12, 6, 6, 6)
        self.local = QCheckBox("Keyboard control")
        self.local.setToolTip("Uncheck to accept commands from an external ROS node.")
        self.local.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.local.setChecked(True)
        self.local.setEnabled(ros)
        self.local.toggled.connect(self.change_authority)
        panel.addWidget(self.local)
        panel.addSpacing(10)
        panel.addWidget(QLabel("Target speed"))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.01, 0.10)
        self.speed.setSingleStep(0.01)
        self.speed.setValue(speed)
        self.speed.setSuffix(" m/s")
        self.speed.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.speed.valueChanged.connect(self.change_speed)
        panel.addWidget(self.speed)
        keys = QGridLayout()
        for row, (a, b, axis) in enumerate((("w", "s", "X"), ("a", "d", "Y"), ("r", "f", "Z"))):
            for col, (key, sign) in enumerate(((a, "+"), (b, "−"))):
                button = QPushButton(f"{sign}{axis}  [{key.upper()}]")
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                button.clicked.connect(lambda checked=False, k=key: self.press(k))
                keys.addWidget(button, row, col)
        panel.addLayout(keys)
        stop = QPushButton("Stop  [Space]")
        stop.setObjectName("stop")
        stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        stop.clicked.connect(lambda: self.press("space"))
        panel.addWidget(stop)
        self.command_label = QLabel()
        self.command_label.setObjectName("muted")
        panel.addWidget(self.command_label)
        panel.addSpacing(12)
        self.phase = QProgressBar()
        self.phase.setRange(0, 100)
        panel.addWidget(self.phase)
        panel.addSpacing(12)
        self.cards = []
        for _ in range(4):
            label = QLabel()
            label.setObjectName("readout")
            self.cards.append(label)
            panel.addWidget(label)
        panel.addStretch()
        trace = QCheckBox("Show path")
        trace.setChecked(True)
        trace.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        trace.toggled.connect(
            lambda checked: setattr(self.view, "show_trail", checked) if self.view else None
        )
        panel.addWidget(trace)
        hint = QLabel("Drag to rotate · Scroll to zoom\nSpace to stop · Q to quit")
        hint.setObjectName("muted")
        panel.addWidget(hint)
        main.addWidget(self.controls)
        splitter.addWidget(stage)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        plots = QWidget()
        grid = QGridLayout(plots)
        grid.setContentsMargins(0, 0, 0, 0)
        self.plots = [
            Plot("Velocity · m/s (body)", ("X", "Y", "Z")),
            Plot("Attitude · deg", ("roll", "pitch", "yaw")),
            Plot("Displacement · m (world)", ("X", "Y", "Z")),
            Plot("Mechanical power · W", ("Σ |τ ω|",)),
        ]
        for i, plot in enumerate(self.plots):
            grid.addWidget(plot, 0, i)
        self.tabs.addTab(plots, "Measurements")
        self.joints = QTableWidget(sim.model.nu, 4)
        self.joints.setHorizontalHeaderLabels(["Joint", "Actual °", "Target °", "Torque N·m"])
        self.joints.horizontalHeader().setStretchLastSection(True)
        self.joints.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(sim.controller.stroke.names):
            for col in range(4):
                self.joints.setItem(row, col, QTableWidgetItem(name if col == 0 else "0"))
        self.tabs.addTab(self.joints, "Joints")
        splitter.setSizes([560, 190])
        self.state_label = QLabel()
        self.statusBar().addPermanentWidget(self.state_label)
        # Window shortcuts also work while the speed editor or joint table has focus.
        self.shortcuts = []
        for sequence, key in [(k.upper(), k) for k in KeyboardCommand.KEYS] + [
            ("Space", "space"),
            ("Q", "q"),
            ("Escape", "escape"),
            ("Tab", "presentation"),
            ("F11", "fullscreen"),
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

    def set_view(self, azimuth, elevation):
        if self.view:
            self.view.camera.azimuth, self.view.camera.elevation = azimuth, elevation

    def set_presentation(self, enabled):
        self.controls.setVisible(not enabled)
        self.tabs.setVisible(not enabled)

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def change_authority(self, local):
        self.command.press("space")
        # Send one zero when releasing local control, then stop publishing.
        self.send_command()

    def change_speed(self, speed):
        self.command.speed = speed
        self.command.velocity[:] = [float(np.sign(v) * speed) for v in self.command.velocity]

    def press(self, key):
        if key == "presentation":
            self.presentation.toggle()
            return
        if key == "fullscreen":
            self.toggle_fullscreen()
            return
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
            f"{'ROS 2' if self.ros else 'Local'}  ·  {s.data.time:.1f} s  ·  {s.data.time / max(elapsed, 0.01):.2f}× realtime"
        )
        self.command_label.setText("XYZ  " + "  ".join(f"{v:+.2f}" for v in s.controller.command))
        self.phase.setValue(round(s.output.phase * 100) if s.output else 0)
        self.phase.setFormat(f"{s.status}  %p%")
        values = [
            f"Travel   {np.linalg.norm(s.displacement) * 100:6.1f} cm",
            f"Speed    {np.linalg.norm(s.velocity) * 100:6.1f} cm/s",
            f"R / P    {s.attitude[0]:+.1f}° / {s.attitude[1]:+.1f}°",
            f"Work     {s.energy:6.2f} J",
        ]
        self.statusBar().showMessage(
            f"X {s.displacement[0]:+.3f} m    Y {s.displacement[1]:+.3f} m    Z {s.displacement[2]:+.3f} m"
        )
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
            self.joints.item(i, 3).setForeground(QColor("#d6a367" if abs(torque) > 1.8 else "#ddd"))

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
