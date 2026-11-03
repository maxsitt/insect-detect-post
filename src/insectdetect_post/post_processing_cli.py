"""Command line interface for running the post-processing pipeline without the GUI.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Run the post-processing pipeline, including image processing, classification,
metadata processing/aggregation, and crop sorting from the terminal, without the GUI.
All settings of the config file are run through the same checks and dataset rules as the GUI
(see pipeline_setup), so both produce the same output for the same config and dataset.

Log messages and progress go to stderr; on success, the path of the run directory is printed
to stdout. In a terminal, the progress is shown in a status line below the log messages,
otherwise (e.g. in the log file of a batch job) as one line per 10 % of each step.

Ctrl+C or SIGTERM cancel the run at the next checkpoint, a second one stops the process immediately.

Exit codes: 0 finished (or check passed), 1 run failed, 2 invalid config or dataset,
130 cancelled with Ctrl+C, 143 terminated with SIGTERM.

Functions:
    main(): Entry point for the 'process' command.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

import yaml

from insectdetect_post.config import AppConfig, load_config_selector, load_config_yaml
from insectdetect_post.constants import CONFIGS_PATH
from insectdetect_post.dataset_context import DatasetContext
from insectdetect_post.exceptions import PipelineCancelled
from insectdetect_post.log_utils import configure_logging
from insectdetect_post.pipeline_runner import PipelineRunner
from insectdetect_post.pipeline_setup import (
    adapt_run_config,
    check_run_config,
    resolve_and_validate_path,
    validate_paths,
)
from insectdetect_post.validation_run import ValidationRun

# Create module-level logger
logger = logging.getLogger(__name__)

# Exit codes; a cancelled run exits with EXIT_SIGNAL_BASE + the signal number (130 for Ctrl+C, 143 for SIGTERM)
EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_INVALID_INPUT = 2
EXIT_SIGNAL_BASE = 128

# Minimum time between two redraws of the status line (seconds)
_STATUS_INTERVAL_S = 0.1

# Shown at the end of '--help'
_HELP_EPILOG = """\
examples:
  uv run --no-sync process
      run the pipeline with settings from the active config file (set in 'configs/config_selector.yaml')
  uv run --no-sync process --config server.yaml --check
      check a config file in 'configs/' and show which steps would run on the dataset, without running them
  uv run --no-sync process --config /path/to/config.yaml --source /path/to/data --output /path/to/output
      run the pipeline with settings from a specified config file and replace the source and output directories

exit codes:
  0 finished (or check passed), 1 run failed, 2 invalid config or dataset,
  130 cancelled with Ctrl+C, 143 terminated with SIGTERM
"""


class _InvalidInput(Exception):
    """The config, its paths or the dataset cannot be used, nothing was started (EXIT_INVALID_INPUT)."""


class _RunFailed(Exception):
    """The inspection or the run failed, the error is already logged with its traceback (EXIT_RUN_FAILED)."""


class _StatusLineStream:
    """Terminal stream with a status line at the bottom, which all other output is printed above.

    Replaces sys.stderr during the run, so it receives the log messages and the direct output of
    other libraries (e.g. warnings, download progress bars): every complete line erases the status
    line, is written, and the status line is drawn again below it. Output that ends without a
    line break (e.g. a progress bar redrawn with '\\r') keeps the bottom line until it is finished.
    Writes can come from several threads, so they share a lock.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._status = ""
        self._line_open = False  # the bottom line holds unfinished output instead of the status
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        """Pass everything else (e.g. isatty(), encoding) on to the terminal stream."""
        return getattr(self._stream, name)

    def write(self, text: str) -> int:
        """Write output above the status line."""
        if not text:
            return 0
        with self._lock:
            if not self._line_open:
                self._erase_status()
            self._stream.write(text)
            self._line_open = not text.endswith("\n")
            if not self._line_open:
                self._stream.write(self._status)
            self._stream.flush()
        return len(text)

    def flush(self) -> None:
        """Flush the terminal stream."""
        self._stream.flush()

    def set_status(self, text: str) -> None:
        """Replace the status line, shortened to the terminal width."""
        text = text[:max(self._columns() - 1, 20)]
        with self._lock:
            if self._line_open:  # keep the unfinished output and draw the status below it
                self._stream.write("\n")
                self._line_open = False
            else:
                self._erase_status()
            self._stream.write(text)
            self._stream.flush()
            self._status = text

    def clear_status(self) -> None:
        """Remove the status line."""
        with self._lock:
            if not self._line_open:
                self._erase_status()
                self._stream.flush()
            self._status = ""

    def _columns(self) -> int:
        """Return the width of the terminal this stream writes to (not of stdout, which may be redirected)."""
        try:
            return os.get_terminal_size(self._stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            return shutil.get_terminal_size().columns

    def _erase_status(self) -> None:
        """Overwrite the status line with spaces (ANSI codes do not work in older Windows consoles)."""
        if self._status:
            self._stream.write("\r" + " " * len(self._status) + "\r")


class _ProgressDisplay:
    """Show the progress of the dataset inspection and of the pipeline steps.

    The runner reports a percentage (set_percent) followed by a message (show_message).
    A message without a new percentage announces the next step, which starts again at 0 %.

    With a terminal, as a status line that is redrawn when the percentage changes, otherwise at
    most every _STATUS_INTERVAL_S. Without one, as one line per 10 % of each step (step starts
    are already logged by the runner).
    """

    def __init__(self, status_line: _StatusLineStream | None, stream: TextIO) -> None:
        self._status_line = status_line
        self._stream = stream
        self._percent: int | None = None
        self._shown_percent: int | None = None
        self._last_redraw = 0.0
        self._last_decile = -1

    def set_percent(self, percent: int) -> None:
        """Store the progress of the current step, shown with the next message."""
        self._percent = percent

    def show_message(self, message: str) -> None:
        """Show a status message, with the percentage reported just before it."""
        percent, self._percent = self._percent, None
        if percent is None:  # the next step starts
            self._last_decile = -1
            self._shown_percent = None
            if self._status_line is not None:
                self._status_line.set_status(f"      {message}")
            return

        if self._status_line is not None:
            now = time.monotonic()
            if percent != self._shown_percent or now - self._last_redraw >= _STATUS_INTERVAL_S:
                self._status_line.set_status(f"{percent:3d}%  {message}")
                self._shown_percent = percent
                self._last_redraw = now
            return

        decile = percent // 10
        if decile > self._last_decile:
            self._last_decile = decile
            self._stream.write(f"{datetime.now():%H:%M:%S} {percent:3d}%  {message}\n")  # noqa: DTZ005
            self._stream.flush()

    def show_progress(self, percent: int, message: str) -> None:
        """Show a status message with its percentage."""
        self.set_percent(percent)
        self.show_message(message)

    def clear(self) -> None:
        """Remove the status line and start counting from 0 again."""
        if self._status_line is not None:
            self._status_line.clear_status()
        self._percent = None
        self._shown_percent = None
        self._last_decile = -1


class _Cancellation:
    """Turn Ctrl+C and SIGTERM into a cancel of the run; a second signal stops the process at once."""

    def __init__(self) -> None:
        self.signum: int | None = None
        self.runner: PipelineRunner | None = None
        self._previous_handlers = {
            signum: signal.signal(signum, self._on_signal) for signum in (signal.SIGINT, signal.SIGTERM)
        }

    @property
    def exit_code(self) -> int:
        """Exit code of a cancelled run (EXIT_SIGNAL_BASE + the signal number)."""
        return EXIT_SIGNAL_BASE + (self.signum or signal.SIGINT)

    def raise_if_cancelled(self) -> None:
        """Raise PipelineCancelled if a signal was received."""
        if self.signum is not None:
            raise PipelineCancelled()

    def restore_handlers(self) -> None:
        """Reinstall the signal handlers that were set before."""
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)

    def _on_signal(self, signum: int, _frame: FrameType | None) -> None:
        """Cancel the run at the first signal, stop the process at the second.

        The handler can interrupt a write to stderr, and writing to the same stream from the
        handler would then fail. So it logs from a new thread, or writes to the file descriptor.
        """
        if self.signum is not None:
            try:
                os.write(2, b"\nStopping immediately\n")
            except OSError:
                pass
            os._exit(EXIT_SIGNAL_BASE + signum)
        self.signum = signum
        if self.runner is not None:
            self.runner.cancel()
        threading.Thread(
            target=logger.warning,
            args=("Cancelling at the next checkpoint (press Ctrl+C again to stop immediately)...",),
            daemon=True,
        ).start()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        prog="process",
        description=("Run the post-processing pipeline with the settings of a config file,\n"
                     "without the GUI (e.g. on a headless server)."),
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", metavar="FILE",
        help="path to a config file, or the file name of a config in 'configs/' (default: the active config file)"
    )
    parser.add_argument(
        "-s", "--source", metavar="DIR",
        help="path to a source directory (replaces 'source_path' of the config)"
    )
    parser.add_argument(
        "-o", "--output", metavar="DIR",
        help="path to an output directory (replaces 'output_path' of the config)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="check the config and the dataset and show which steps would run, without running them"
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="only show warnings and errors")
    verbosity.add_argument("-v", "--verbose", action="store_true", help="also show debug messages")
    return parser.parse_args(argv)


def _resolve_config_path(config_arg: str | None) -> Path:
    """Return the config file to use: the given path, a config file in 'configs/' or the active config file.

    Raises:
        _InvalidInput: If the given config file does not exist or the active config file cannot be read.
    """
    if config_arg is None:
        try:
            return CONFIGS_PATH / load_config_selector().config_active
        except OSError as e:
            raise _InvalidInput(f"Cannot read the active config file in '{CONFIGS_PATH}': {e}") from e

    for candidate in (Path(config_arg).expanduser(), CONFIGS_PATH / config_arg):
        if candidate.is_file():
            return candidate.resolve()
    raise _InvalidInput(f"Config file not found: '{config_arg}' (neither a path nor a file name in '{CONFIGS_PATH}')")


def _load_config(args: argparse.Namespace) -> tuple[AppConfig, Path]:
    """Load the config (never changing the file) and apply overrides from the command line arguments.

    Returns:
        The config with the applied overrides and the path of its file.

    Raises:
        _InvalidInput: If the config file cannot be read or is invalid.
    """
    config_path = _resolve_config_path(args.config)
    logger.info("Config: %s", config_path)
    try:
        config = load_config_yaml(config_path, write_back=False, strict=True)
    except (OSError, yaml.YAMLError, ValueError) as e:
        raise _InvalidInput(str(e)) from e

    overrides = {key: value for key, value in (("source_path", args.source), ("output_path", args.output))
                 if value is not None}
    for key, value in overrides.items():
        logger.info("Using '%s' from the command line: %s", key, value)
    if overrides:
        config = AppConfig.model_validate({**config.model_dump(), **overrides})
    return config, config_path


def _prepare_run(
    args: argparse.Namespace,
    display: _ProgressDisplay | None,
    cancellation: _Cancellation
) -> tuple[PipelineRunner, DatasetContext]:
    """Load the config, inspect the dataset and apply the same checks and dataset rules as the GUI.

    Returns:
        The runner for the adapted config, and the dataset inspection.

    Raises:
        _InvalidInput: If the config, its paths or the dataset cannot be used.
        _RunFailed: If the dataset inspection failed.
        PipelineCancelled: If a signal was received during the inspection.
    """
    config, config_path = _load_config(args)

    error = validate_paths(config.source_path, config.output_path)
    if error:
        raise _InvalidInput(error.replace("\n", " "))
    source_dir = resolve_and_validate_path(config.source_path)
    if source_dir is None:
        raise _InvalidInput(f"Invalid source directory: {config.source_path}")

    logger.info("Inspecting dataset: %s", source_dir)

    def on_inspection_progress(percent: int, message: str) -> None:
        cancellation.raise_if_cancelled()
        if display is not None:
            display.show_progress(percent, message)

    try:
        context = DatasetContext.from_inspection(source_dir, progress_callback=on_inspection_progress)
    except PipelineCancelled:
        raise
    except Exception as e:
        logger.exception("Dataset inspection failed")
        raise _RunFailed(str(e)) from e
    finally:
        if display is not None:
            display.clear()

    config, changes = adapt_run_config(config, context)
    if changes:
        logger.warning("Settings adjusted to the dataset before the run:\n  %s", "\n  ".join(changes))
    error = check_run_config(config, context)
    if error:
        raise _InvalidInput(error.replace("\n", " "))

    runner = PipelineRunner(
        config,
        config_stem=config_path.stem,
        context=context,
        on_progress=display.set_percent if display is not None else None,
        on_message=display.show_message if display is not None else None,
    )
    return runner, context


def _run_pipeline(runner: PipelineRunner, display: _ProgressDisplay | None, cancellation: _Cancellation) -> Path:
    """Run the pipeline, cancellable with Ctrl+C or SIGTERM.

    Returns:
        The path to the run directory.

    Raises:
        _RunFailed: If the run failed (the runner has logged the error).
        PipelineCancelled: If a signal was received.
    """
    cancellation.runner = runner
    cancellation.raise_if_cancelled()  # a signal during the preparation
    try:
        return runner.run()
    except PipelineCancelled:
        raise
    except Exception as e:
        raise _RunFailed(str(e)) from e
    finally:
        if display is not None:
            display.clear()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the 'process' command.

    Args:
        argv: Command line arguments (default: sys.argv[1:]).

    Returns:
        Exit code.
    """
    args = _parse_args(argv)
    # Like stderr, write characters the output encoding lacks as escapes instead of failing
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="backslashreplace")
    console_level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    stderr = sys.stderr
    status_line = _StatusLineStream(stderr) if stderr.isatty() and not args.quiet else None
    if status_line is not None:
        # Direct output of other libraries (e.g. warnings, download progress bars) also goes above the status line
        sys.stderr = status_line  # type: ignore[assignment]
    configure_logging(console_level=console_level, stream=status_line)
    display = None if args.quiet else _ProgressDisplay(status_line, stderr)
    cancellation = _Cancellation()

    try:
        runner, context = _prepare_run(args, display, cancellation)
        if args.check:
            logger.info("\n%s", context.get_summary_str(include_details=True))
            steps = runner.planned_steps()
            logger.info("Steps that would run: %s", ", ".join(steps) if steps else "none")
            logger.info("Check passed - nothing was run")
            return EXIT_OK

        run_dir = _run_pipeline(runner, display, cancellation)
        print(run_dir, flush=True)  # stdout, e.g. for scripts
        if ValidationRun.find_final_csv(run_dir) is not None:
            logger.info("Validate the results in the validation GUI with 'uv run --no-sync validate'")
        return EXIT_OK

    except _InvalidInput as e:
        logger.error("%s", e)
        return EXIT_INVALID_INPUT
    except PipelineCancelled:
        logger.warning("Cancelled")
        return cancellation.exit_code
    except _RunFailed as e:
        logger.error("Run failed: %s", e)
        return EXIT_RUN_FAILED
    except Exception:
        logger.exception("Unexpected error")
        return EXIT_RUN_FAILED
    finally:
        cancellation.restore_handlers()
        sys.stderr = stderr


if __name__ == "__main__":
    sys.exit(main())
