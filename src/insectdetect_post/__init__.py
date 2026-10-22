"""insect-detect-post - Software for post-processing of camera trap data.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

This package provides post-processing tools for Insect Detect camera trap data,
including image processing, classification, metadata filtering, and validation.
"""

import os

# Keep ultralytics from pip-installing packages at runtime, which would bypass uv.lock
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
