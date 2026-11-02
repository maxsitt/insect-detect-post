"""Run dataset inspection in a cancellable background thread and report progress via Qt signals.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Classes:
    DatasetInspector: Background worker for inspecting a source dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from insectdetect_post.dataset_context import DatasetContext
from insectdetect_post.exceptions import PipelineCancelled

# Create module-level logger
logger = logging.getLogger(__name__)


class DatasetInspector(QObject):
    """Background worker for inspecting a source dataset."""
    progress = Signal(int)
    progress_message = Signal(str)
    finished = Signal(DatasetContext)
    error = Signal(str)

    def __init__(self, root_path: Path) -> None:
        super().__init__()
        self.root_path = root_path
        self._cancelled = False

    def run(self) -> None:
        """Execute dataset inspection and create dataset context."""
        try:
            def progress_cb(pct: int, msg: str) -> None:
                if self._cancelled:
                    raise PipelineCancelled("Dataset inspection cancelled")
                self.progress.emit(pct)
                self.progress_message.emit(msg)

            context = DatasetContext.from_inspection(
                self.root_path,
                progress_callback=progress_cb
            )

            if self._cancelled:
                return

            self.progress.emit(100)
            self.finished.emit(context)
        except PipelineCancelled:
            logger.info("Dataset inspection cancelled")
        except Exception as e:
            logger.exception("Dataset inspection failed")
            self.error.emit(f"Inspection failed: {e}")

    def cancel(self) -> None:
        """Request cancellation of dataset inspection."""
        if not self._cancelled:
            self._cancelled = True
            logger.info("Dataset inspection cancellation requested")
