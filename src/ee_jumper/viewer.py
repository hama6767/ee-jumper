"""Wait for MuJoCo's asynchronous renderer to release GLFW before exit."""

import threading


class PassiveViewer:
    def __init__(self, model, data):
        from mujoco import viewer

        before = set(threading.enumerate())
        self.handle = viewer.launch_passive(model, data)
        # Linux/Windows MuJoCo launches a daemon named after this target.
        # On macOS mjpython owns the UI thread, so there is nothing to join.
        self.render_threads = [
            t
            for t in threading.enumerate()
            if t not in before and t.name.endswith("(_launch_internal)")
        ]

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def close(self):
        self.handle.close()
        # Handle.close() only requests exit. Wait before atexit terminates GLFW.
        for thread in self.render_threads:
            thread.join(timeout=5)
