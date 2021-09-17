from typing import NamedTuple, Optional

import numpy as np


__all__ = ['ClassificationOutput', 'TranslationOutput']


class ClassificationOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    probs: Optional[np.ndarray]
    metrics: Optional[dict]


class TranslationOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    probs: Optional[np.ndarray]
    metrics: Optional[dict]
