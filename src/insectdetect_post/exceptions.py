"""Custom exceptions for the insect-detect-post modules.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

Classes:
    PipelineCancelled:  Raised when the user requests cancellation of a pipeline or inspection run.
    UpdateError:        Raised when an update step fails and needs manual intervention.
    ValidationRunError: Raised when a run directory cannot be used for validation.
    RunSearchCancelled: Raised when the search for a run to validate was cancelled.
"""

from __future__ import annotations


class PipelineCancelled(Exception):
    """Raised when the user requests cancellation of a pipeline or inspection run."""


class UpdateError(Exception):
    """Raised when an update step fails and needs manual intervention."""


class ValidationRunError(Exception):
    """Raised when a run directory cannot be used for validation."""


class RunSearchCancelled(Exception):
    """Raised when the search for a run to validate was cancelled."""
