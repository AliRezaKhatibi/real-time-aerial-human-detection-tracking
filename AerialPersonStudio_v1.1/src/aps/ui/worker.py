from __future__ import annotations
from PySide6.QtCore import QObject, Signal, Slot
import traceback


class FunctionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)
    log = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self):
        try:
            result = self.fn(self.progress.emit, self.log.emit)
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
