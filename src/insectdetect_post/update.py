"""Update the local installation to the latest version available on GitHub.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Shows the available updates, fast-forwards the local repository to the latest version
and re-syncs the Python packages if the dependencies changed. Config files are not
tracked by git and are therefore never modified or removed by an update.

Classes:
    UpdateInfo: Available update with its incoming commits, file changes and version bump.

Functions:
    check_for_updates(): Fetch latest changes from GitHub and return the available update, if any.
    apply_update():      Fast-forward to the latest version and sync the dependencies.
    sync_command():      Return the command that syncs the packages, keeping the installed extra.
    run_update():        Show the available update, ask for confirmation and apply it.
    main():              Entry point for updating from the command line.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distributions
from importlib.metadata import version as installed_version

from insectdetect_post.constants import BASE_PATH
from insectdetect_post.exceptions import UpdateError

# Create module-level logger
logger = logging.getLogger(__name__)

# Remote branch that updates are fetched from and fast-forwarded to
REMOTE_NAME = "origin"
MAIN_BRANCH = "main"
REMOTE_BRANCH = f"{REMOTE_NAME}/{MAIN_BRANCH}"

# Timeout in seconds for a single git command, to never block a caller indefinitely
GIT_TIMEOUT = 120

# Files that require re-syncing the Python packages if they changed upstream
SYNC_TRIGGER_FILES = ("pyproject.toml", "uv.lock", ".python-version")

# Git diff status codes, mapped to the label used when listing the incoming changes
FILE_STATUS_LABELS = {"A": "added", "C": "copied", "D": "removed",
                      "M": "updated", "R": "renamed", "T": "updated"}


@dataclass(frozen=True)
class UpdateInfo:
    """Available update with its incoming commits, file changes and version bump."""
    commits: list[str]
    changes: list[tuple[str, str]]
    local_changes: list[str]
    version_local: str | None
    version_remote: str | None

    @property
    def needs_sync(self) -> bool:
        """Check if the update changes dependency files and requires a package sync."""
        return any(path in SYNC_TRIGGER_FILES for _, path in self.changes)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command scoped to BASE_PATH and return the completed process.

    Raises:
        UpdateError: If the command times out, or if check is True and it exits
                     with a non-zero status.
    """
    try:
        result = subprocess.run(
            # Git writes UTF-8 and quotes non-ASCII paths unless told otherwise
            ["git", "-C", str(BASE_PATH), "-c", "core.quotePath=false", *args],
            capture_output=True, encoding="utf-8", errors="replace", check=False, timeout=GIT_TIMEOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},  # never wait for a credential prompt
        )
    except subprocess.TimeoutExpired as e:
        raise UpdateError(f"'git {' '.join(args)}' timed out after {GIT_TIMEOUT} seconds.") from e

    if check and result.returncode != 0:
        raise UpdateError(f"'git {' '.join(args)}' failed:\n{result.stderr.strip()}")
    return result


def _confirm(prompt: str) -> bool:
    """Ask a yes/no question on stdin, defaulting to no (also if stdin is not interactive)."""
    try:
        return input(f"{prompt} (y/N): ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _check_prerequisites() -> None:
    """Verify that git/uv are available and the repository is in an updatable state.

    Raises:
        UpdateError: If a required tool is missing, BASE_PATH is not a git repository
                     with an 'origin' remote, or the active branch is not the main branch.
    """
    for tool in ("git", "uv"):
        if shutil.which(tool) is None:
            raise UpdateError(f"'{tool}' is required but was not found on PATH.")

    if _git("rev-parse", "--git-dir", check=False).returncode != 0:
        raise UpdateError(f"'{BASE_PATH}' is not a git repository.")

    if _git("remote", "get-url", REMOTE_NAME, check=False).returncode != 0:
        raise UpdateError(f"No '{REMOTE_NAME}' remote is configured in '{BASE_PATH}'.")

    branch = _git("symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if branch != MAIN_BRANCH:
        raise UpdateError(
            f"Updates can only be applied on the '{MAIN_BRANCH}' branch (currently on "
            f"'{branch or 'a detached HEAD'}').\nRun 'git checkout {MAIN_BRANCH}' and try again."
        )


def _local_changes() -> list[str]:
    """Return all tracked files with uncommitted local changes."""
    return [line[3:] for line in
            _git("status", "--porcelain", "--untracked-files=no").stdout.splitlines() if line]


def _get_version(pyproject: str) -> str | None:
    """Extract the project version from the content of a pyproject.toml file."""
    try:
        return tomllib.loads(pyproject)["project"]["version"]
    except (tomllib.TOMLDecodeError, KeyError, TypeError):
        return None


def _install_extras() -> dict[str, str]:
    """Map each install extra to the local version suffix of its torch build.

    Derived from the PyTorch indexes in pyproject.toml, whose last URL path segment is the
    local version label of the wheels they serve ('.../whl/cu132' -> '2.13.0+cu132').

    Returns:
        Mapping of extra name to torch version suffix, empty if it could not be derived.
    """
    try:
        pyproject = tomllib.loads((BASE_PATH / "pyproject.toml").read_text(encoding="utf-8"))
        uv_config = pyproject["tool"]["uv"]
        index_urls = {index["name"]: index["url"] for index in uv_config["index"]}
        return {source["extra"]: "+" + index_urls[source["index"]].rstrip("/").rsplit("/", 1)[-1]
                for source in uv_config["sources"]["torch"]}
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as e:
        logger.debug("Could not derive the install extras from pyproject.toml (%s).", e)
        return {}


def _detect_install_extra() -> str | None:
    """Detect which of the mutually exclusive extras is installed.

    Reads the local version suffix off the installed torch build (e.g. '+cu132', '+cpu').
    Builds without a local version suffix are resolved via the installed ONNX runtime variant.

    Returns:
        The matching extra name, or None if it could not be determined.
    """
    try:
        torch_version = installed_version("torch")
    except PackageNotFoundError:
        return None

    extras = _install_extras()
    for extra, suffix in extras.items():
        if torch_version.endswith(suffix):
            return extra

    # Without a local version suffix, only the CPU variant can be identified unambiguously
    packages = {dist.metadata["Name"] for dist in distributions() if dist.metadata["Name"]}
    if "onnxruntime" not in packages:
        return None
    return next((extra for extra, suffix in extras.items() if suffix == "+cpu"), None)


def _sync_dependencies() -> None:
    """Re-sync the Python packages, preserving the installed extra."""
    logger.info("Dependency files changed. Syncing packages...")

    # Never sync without an extra, as this would uninstall PyTorch and the ONNX runtime
    extra = _detect_install_extra()
    if extra is None:
        logger.warning("Could not detect the installed version. Run '%s' manually "
                       "to finish the update.", sync_command())
        return

    args = ["uv", "sync", "--extra", extra]
    result = subprocess.run(args, cwd=BASE_PATH, capture_output=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        logger.warning(
            "Failed to sync dependencies automatically:\n%s\nRun '%s' manually.",
            result.stderr.strip(), " ".join(args)
        )
    else:
        logger.info("Dependencies synced (extra: %s).", extra)


def _merge_error(git_error: str) -> UpdateError:
    """Compose the error for a failed fast-forward, with a hint if the branch has own commits."""
    message = ("Could not fast-forward to the latest version, your installation was left "
               f"unchanged:\n{git_error}")

    own_commits = _git("rev-list", "--count", f"{REMOTE_BRANCH}..HEAD", check=False).stdout.strip()
    if own_commits.isdigit() and int(own_commits) > 0:
        message += (f"\nThe local repository contains {own_commits} own commit(s). Resolve the "
                    f"divergence manually, or run 'git reset --hard {REMOTE_BRANCH}' to discard "
                    "them (this cannot be undone).")

    return UpdateError(message)


def check_for_updates() -> UpdateInfo | None:
    """Fetch latest changes from GitHub and return the available update, if any.

    Returns:
        UpdateInfo with the incoming commits and file changes, or None if up to date.

    Raises:
        UpdateError: If the repository is not in an updatable state or a git command fails.
    """
    _check_prerequisites()
    _git("fetch", REMOTE_NAME)

    commits = [line for line in _git("log", "--oneline", f"HEAD..{REMOTE_BRANCH}").stdout.splitlines() if line]
    if not commits:
        return None

    changes = []
    for line in _git("diff", "--name-status", f"HEAD...{REMOTE_BRANCH}").stdout.splitlines():
        if line:
            parts = line.split("\t")
            changes.append((FILE_STATUS_LABELS.get(parts[0][0], "updated"), parts[-1]))

    return UpdateInfo(
        commits=commits,
        changes=changes,
        local_changes=_local_changes(),
        version_local=_get_version((BASE_PATH / "pyproject.toml").read_text(encoding="utf-8")),
        version_remote=_get_version(_git("show", f"{REMOTE_BRANCH}:pyproject.toml").stdout),
    )


def apply_update(info: UpdateInfo, sync: bool = True) -> None:
    """Fast-forward to the latest version and sync the dependencies.

    Args:
        info: Available update, as returned by check_for_updates().
        sync: Re-sync the Python packages if the update changed the dependency files.

    Raises:
        UpdateError: If the update could not be applied or local changes were left stashed.
    """
    local_changes = _local_changes()
    if local_changes:
        logger.warning("Stashing your local changes to the following file(s):")
        for local_change in local_changes:
            logger.warning("  %s", local_change)
        _git("stash", "push", "-m", "insectdetect-post update: local changes backup")

    logger.info("Applying updates...")
    merge_result = _git("merge", "--ff-only", REMOTE_BRANCH, check=False)
    if merge_result.returncode != 0:
        if local_changes:
            _git("stash", "pop", check=False)
        raise _merge_error(merge_result.stderr.strip())

    if local_changes and _git("stash", "pop", check=False).returncode != 0:
        raise UpdateError(
            "Update applied, but your local changes could not be restored automatically.\n"
            "They are still safely kept in the stash -- run 'git stash list' and "
            "'git stash pop' to resolve the conflict manually."
        )

    if sync and info.needs_sync:
        _sync_dependencies()


def sync_command() -> str:
    """Return the command that syncs the packages, keeping the installed extra."""
    extra = _detect_install_extra() or f"<{'|'.join(_install_extras()) or 'extra'}>"
    return f"uv sync --extra {extra}"


def run_update(confirm: Callable[[str], bool] = _confirm) -> None:
    """Show the available update, ask for confirmation and apply it.

    Args:
        confirm: Callable used to ask for confirmation before the update is applied.

    Raises:
        UpdateError: If a step fails in a way that needs manual intervention.
    """
    logger.info("Checking for updates on GitHub...")
    info = check_for_updates()
    if info is None:
        logger.info("No updates available. Your installation is up to date.")
        return

    if info.version_local and info.version_remote and info.version_local != info.version_remote:
        logger.info("New version available: %s -> %s", info.version_local, info.version_remote)

    logger.info("%d new commit(s):", len(info.commits))
    for commit in info.commits:
        logger.info("  %s", commit)

    logger.info("%d file(s) will be changed:", len(info.changes))
    for status, path in info.changes:
        logger.info("  %-8s %s", status, path)

    if info.local_changes:
        logger.warning("Your local changes to %d file(s) will be stashed and restored afterwards.",
                       len(info.local_changes))

    if not confirm("Apply these updates?"):
        logger.info("Update cancelled.")
        return

    apply_update(info)
    logger.info("Update complete!")


def main() -> None:
    """Entry point for updating from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger.info("==== Insect Detect Post Updater ====\n")

    try:
        run_update()
    except (UpdateError, OSError) as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.error("\nUpdate cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
