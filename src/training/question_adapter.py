"""Adapt video-oriented question text for single/multi-image training.

Usage:
    from training.question_adapter import adapt_question_for_image

    image_q = adapt_question_for_image(video_question)

Replaces "in this video" with "in this image" so the prompt matches the
modality the model receives during image-based SFT.
"""

from __future__ import annotations

import re

_VIDEO_TO_IMAGE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bin this video\b", re.IGNORECASE), "in this image"),
]


def adapt_question_for_image(question: str) -> str:
    """Return *question* with video references replaced by image references."""
    result = question
    for pattern, replacement in _VIDEO_TO_IMAGE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
