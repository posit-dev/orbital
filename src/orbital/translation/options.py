"""Translation options used to customize the ONNX→SQL conversion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TranslationOptions:
    """Configuration knobs that affect translation behaviour."""

    allow_text_tensors: bool = False
