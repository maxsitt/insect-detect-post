"""Run the post-processing pipeline in a background thread and report progress via Qt signals.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Classes:
    PipelineWorker: Qt wrapper that runs a PipelineRunner and turns its callbacks and result into signals.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from insectdetect_post.exceptions import PipelineCancelled
from insectdetect_post.pipeline_runner import PipelineRunner


class PipelineWorker(QObject):
    """Qt wrapper that runs a PipelineRunner and turns its callbacks and result into signals.

    Moved to a QThread by the GUI, so the signals reach the GUI thread as queued connections.
    'error' carries "Pipeline cancelled" for a cancelled run and the error message otherwise.
    """
    progress = Signal(int)
    progress_message = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, runner: PipelineRunner) -> None:
        super().__init__()
        self._runner = runner
        runner.on_progress = self.progress.emit
        runner.on_message = self.progress_message.emit

    @property
    def output_root(self) -> Path:
        """Root output directory of the run; raises if not yet created."""
        return self._runner.output_root

    def cancel(self) -> None:
        """Request cancellation of the pipeline."""
        self._runner.cancel()

    @Slot()
    def run(self) -> None:
        """Run the pipeline and emit 'finished' or 'error' (the runner logs the details)."""
        try:
            self._runner.run()
        except PipelineCancelled:
            self.error.emit("Pipeline cancelled")
        except Exception as e:  # noqa: BLE001 - the runner has logged it with its traceback
            self.error.emit(str(e))
        else:
            self.finished.emit()
