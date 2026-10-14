"""Update button shared by the processing and validation GUI.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Hidden until a background check finds an update. If an update is available and the button is clicked,
the incoming commits are shown, installed if confirmed, and a restart is offered afterwards.

Classes:
    UpdateButton: Button that checks for, shows and installs an available update.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable, Sequence

from PySide6.QtCore import QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from insectdetect_post.constants import BASE_PATH, COLOR_BTN_UPDATE
from insectdetect_post.styled_button import StyledButton
from insectdetect_post.update import UpdateInfo, sync_command
from insectdetect_post.update_runner import UpdateChecker, UpdateInstaller

# Create module-level logger
logger = logging.getLogger(__name__)

# Delay of the initial check, so the window is shown first
_CHECK_DELAY_MS = 1000


class UpdateButton(StyledButton):
    """Button that checks for, shows and installs an available update.

    Disabled while the host window is busy (see set_busy()).

    Signals:
        install_started(): Installation began, the host should lock its controls.
        install_finished(str): Installation ended with a status message, the host should unlock.
    """
    install_started = Signal()
    install_finished = Signal(str)

    def __init__(self, restart_args: Callable[[], Sequence[str]], parent: QWidget | None = None) -> None:
        """Initialize the update button.

        Args:
            restart_args: Called on restart; returns the Python arguments that start the host
                window again (e.g. ['-m', 'insectdetect_post.post_processing_gui']).
            parent: Parent widget.
        """
        super().__init__("Update available", COLOR_BTN_UPDATE)
        if parent is not None:
            self.setParent(parent)
        self.setToolTip("Install the latest version from GitHub")
        self.setVisible(False)
        self.setEnabled(False)
        self._restart_args = restart_args
        self._thread: QThread | None = None
        self._worker: UpdateChecker | UpdateInstaller | None = None
        self._info: UpdateInfo | None = None
        self._available = False
        self._busy = False
        self.clicked.connect(self._on_click)

    @property
    def is_installing(self) -> bool:
        """Whether an installation is running."""
        return isinstance(self._worker, UpdateInstaller)

    def start_check(self, delay_ms: int = _CHECK_DELAY_MS) -> None:
        """Check GitHub for an update in the background after 'delay_ms'."""
        QTimer.singleShot(delay_ms, self._start_check)

    def set_busy(self, busy: bool) -> None:
        """Block the update while the host window runs a task."""
        self._busy = busy
        self._sync_state()

    def shutdown(self) -> None:
        """Release the worker thread (when the host window closes)."""
        self._cleanup_thread()

    def _set_available(self, available: bool) -> None:
        """Show or hide the button."""
        self._available = available
        self.setVisible(available)
        self._sync_state()

    def _sync_state(self) -> None:
        """Enable the button only if an update is available and nothing is running."""
        self.setEnabled(self._available and not self._busy and not self.is_installing)

    def _start_worker(self, worker: UpdateChecker | UpdateInstaller) -> None:
        """Move an update worker to its own thread and start it."""
        thread = QThread()
        self._thread = thread
        self._worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.start()

    def _cleanup_thread(self) -> None:
        """Stop and delete the update thread and worker."""
        if self._thread:
            self._thread.quit()
            if not self._thread.wait(5000):
                logger.warning("Update thread not responding, forcing termination...")
                self._thread.terminate()
                self._thread.wait(500)
            self._thread.deleteLater()
            self._thread = None

        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    @Slot()
    def _start_check(self) -> None:
        """Check for available updates in the background."""
        checker = UpdateChecker()
        checker.finished.connect(self._on_check_finish)
        checker.error.connect(self._on_check_error)
        self._start_worker(checker)

    @Slot(object)
    def _on_check_finish(self, info: UpdateInfo | None) -> None:
        """Show the button if an update is available."""
        self._cleanup_thread()
        self._info = info
        if info is None:
            logger.debug("No updates available")
            return

        logger.info("Update available: %d new commit(s) on GitHub", len(info.commits))
        self._set_available(True)

    @Slot(str)
    def _on_check_error(self, error_msg: str) -> None:
        """Log a failed update check."""
        self._cleanup_thread()
        logger.debug("Update check skipped: %s", error_msg)

    @Slot()
    def _on_click(self) -> None:
        """Show the available update and install it if confirmed."""
        info = self._info
        if info is None:
            return

        if info.version_local and info.version_remote and info.version_local != info.version_remote:
            text = f"Insect Detect Post {info.version_local} → {info.version_remote}"
        else:
            text = "A new version of Insect Detect Post is available."

        informative = (f"{len(info.commits)} new commit(s) will be applied.\n\n"
                       "Your configuration and profiles are not affected.")
        if info.local_changes:
            informative += (f"\n\nYour local changes to {len(info.local_changes)} file(s) will be "
                            "stashed and restored afterwards.")

        details = "\n".join([*info.commits, "",
                             *(f"{status:<8} {path}" for status, path in info.changes)])

        msg = QMessageBox(self.window())
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Update Available")
        msg.setText(text)
        msg.setInformativeText(informative)
        msg.setDetailedText(details)
        install_btn = msg.addButton("Install Update", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(install_btn)
        msg.exec()

        if msg.clickedButton() is install_btn:
            self._install(info)

    def _install(self, info: UpdateInfo) -> None:
        """Install the update in the background."""
        logger.info("Installing update...")
        installer = UpdateInstaller(info)
        installer.finished.connect(self._on_install_finish)
        installer.error.connect(self._on_install_error)
        self._start_worker(installer)
        self._sync_state()
        self.install_started.emit()

    @Slot()
    def _on_install_finish(self) -> None:
        """Prompt for a restart after the update was installed."""
        self._cleanup_thread()
        self._set_available(False)
        self.install_finished.emit("Update installed")
        logger.info("Update complete!")

        info, self._info = self._info, None
        version = f" to version {info.version_remote}" if info and info.version_remote else ""

        msg = QMessageBox(self.window())
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Update Installed")
        msg.setText(f"Insect Detect Post was updated{version}.")

        if info and info.needs_sync:
            command = sync_command()
            msg.setInformativeText(
                "This update changed the dependencies. Close the application and run the "
                f"following command to finish the update:\n\n{command}"
            )
            copy_btn = msg.addButton("Copy Command", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("Close", QMessageBox.ButtonRole.RejectRole)
            msg.exec()
            if msg.clickedButton() is copy_btn:
                QApplication.clipboard().setText(command)
                logger.info("Copied '%s' to the clipboard", command)
        else:
            msg.setInformativeText("Restart the application to use the new version.")
            restart_btn = msg.addButton("Restart Now", QMessageBox.ButtonRole.AcceptRole)
            msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(restart_btn)
            msg.exec()
            if msg.clickedButton() is restart_btn:
                self._restart()

    @Slot(str)
    def _on_install_error(self, error_msg: str) -> None:
        """Report a failed update installation."""
        self._cleanup_thread()
        self._sync_state()
        self.install_finished.emit("Update failed")
        logger.error(error_msg)
        QMessageBox.critical(self.window(), "Update Failed", error_msg)

    def _restart(self) -> None:
        """Start a new instance of the application and close the current one."""
        window = self.window()
        try:
            process = subprocess.Popen([sys.executable, *self._restart_args()], cwd=BASE_PATH)
        except OSError:
            logger.exception("Could not restart the application")
            QMessageBox.warning(window, "Restart Failed",
                                "Could not restart automatically. Please start the application again.")
            return

        # Stop the new instance again if closing the current one was cancelled
        if not window.close():
            logger.info("Restart cancelled.")
            process.terminate()
