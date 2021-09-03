from typing import NamedTuple, Optional

import numpy as np


class TranslationOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    probs: Optional[np.ndarray]


class ClassificationOutput(NamedTuple):
    """Output object returned by `transformer.predict` method."""
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    probs: Optional[np.ndarray]
