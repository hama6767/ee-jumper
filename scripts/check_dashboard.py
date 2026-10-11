"""Exercise the real Qt/MuJoCo window, keyboard, recovery, CSV and clean close."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from ee_jumper.dashboard import Dashboard
from ee_jumper.learn_cycles import load_library
from ee_jumper.simulation import Swimmer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/dashboard-check.json")
    parser.add_argument("--screenshot", default="runs/dashboard.png")
    args = parser.parse_args()
    app = QApplication([])
    sim = Swimmer(load_library("examples/strokes-xyz.json"))
    window = Dashboard(sim)
    window.show()
    window.activateWindow()
    phase, result, errors = ["idle"], {}, []
    start = time.monotonic()
    initial = sim.data.qpos.copy()

    def check():
        try:
            assert time.monotonic() - start < 90, "GUI stalled"
            if phase[0] == "idle" and sim.data.time >= 0.5:
                np.testing.assert_allclose(sim.data.qpos, initial, atol=1e-10)
                result["idle_drift_m"] = float(np.linalg.norm(sim.displacement))
                assert window.windowTitle() == "Jumper"
                # Native shortcuts must work even when a numeric editor has focus.
                window.speed.setFocus()
                QTest.keyClick(window.speed, Qt.Key.Key_Tab)
                assert window.presentation.isChecked()
                assert not window.controls.isVisible() and not window.tabs.isVisible()
                QTest.keyClick(window, Qt.Key.Key_F11)
                assert window.isFullScreen()
                QTest.keyClick(window, Qt.Key.Key_F11)
                QTest.keyClick(window, Qt.Key.Key_Tab)
                assert window.controls.isVisible() and window.tabs.isVisible()
                window.set_view(90, -89)
                assert window.view.camera.elevation == -89
                window.set_view(125, -27)
                result["presentation_fullscreen_camera"] = True
                window.reset(25)
                phase[0] = "recovery"
            elif phase[0] == "recovery" and sim.data.time >= 10:
                result["roll_after_10s_deg"] = float(sim.attitude[0])
                assert abs(sim.attitude[0]) < 1
                window.reset()
                QTest.keyClick(window, Qt.Key.Key_W)
                phase[0] = "swim"
            elif phase[0] == "swim" and sim.data.time >= 4:
                assert sim.displacement[0] > 0.10, sim.displacement
                result["x_progress_4s_m"] = float(sim.displacement[0])
                assert sim.energy > 0
                result["mechanical_work_j"] = sim.energy
                image = Path(args.screenshot)
                image.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(image))
                assert len(window.view.trail) > 10
                QTest.keyClick(window, Qt.Key.Key_Tab)
                app.processEvents()
                window.refresh()
                window.grab().save(str(image.with_stem(image.stem + "-presentation")))
                # Use the real exporter, choosing a deterministic path without a modal dialog.
                csv_path = Path("runs/dashboard-check.csv")
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                original_dialog = QFileDialog.getSaveFileName
                try:
                    QFileDialog.getSaveFileName = lambda *a: (str(csv_path), "CSV (*.csv)")
                    window.export_csv()
                finally:
                    QFileDialog.getSaveFileName = original_dialog
                assert len(csv_path.read_text().splitlines()) > 20
                QTest.keyClick(window, Qt.Key.Key_Space)
                phase[0] = "stop"
            elif phase[0] == "stop" and sim.data.time >= 4.7:
                assert sim.status == "idle"
                np.testing.assert_allclose(sim.data.ctrl, sim.controller.stroke.home, atol=1e-10)
                result["space_stops_stroking"] = True
                result["stop_works_in_presentation"] = window.presentation.isChecked()
                window.reset()
                np.testing.assert_allclose(sim.displacement, 0)
                assert sim.energy == 0
                result["reset_clears_metrics"] = True
                timer.stop()
                QTest.keyClick(window, Qt.Key.Key_Q)
        except Exception:
            errors.append(traceback.format_exc())
            timer.stop()
            window.close()

    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(50)
    app.exec()
    error = window.error
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not errors and error is None, (errors, error)
    result.update(
        passed=True,
        real_pyside6_window=True,
        embedded_mujoco=True,
        wall_seconds=time.monotonic() - start,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
