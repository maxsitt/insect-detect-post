"""Check for and apply updates in the background, without blocking the GUI.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Wraps the update logic in Qt workers that run in a separate thread and report
their result via signals, so the GUI stays responsive while git is running.

Classes:
    UpdateChecker: Checks for an available update and reports the result.
    UpdateInstaller: Applies an available update and reports the result.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, Slot

from insectdetect_post.exceptions import UpdateError
from insectdetect_post.update import UpdateInfo, apply_update, check_for_updates

# Create module-level logger
logger = logging.getLogger(__name__)


class UpdateChecker(QObject):
    """Checks for an available update and reports the result."""
    finished = Signal(object)
    error = Signal(str)

    @Slot()
    def run(self) -> None:
        """Check GitHub for an available update."""
        try:
            self.finished.emit(check_for_updates())
        except (UpdateError, OSError) as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.exception("Update check failed")
            self.error.emit(str(e))


class UpdateInstaller(QObject):
    """Applies an available update and reports the result."""
    finished = Signal()
    error = Signal(str)

    def __init__(self, info: UpdateInfo) -> None:
        """Initialize the installer with the update to apply."""
        super().__init__()
        self._info = info

    @Slot()
    def run(self) -> None:
        """Apply the update, leaving the packages untouched while they are in use."""
        try:
            apply_update(self._info, sync=False)
            self.finished.emit()
        except (UpdateError, OSError) as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.exception("Update installation failed")
            self.error.emit(str(e))
