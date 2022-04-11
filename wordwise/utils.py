from typing import Callable, List, Tuple, Union

import torch
from sklearn.feature_extraction.text import CountVectorizer


def squash(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(
            f"Expected `torch.Tensor`, but got an unexpected `value` of type {value.__class__}"
        )
    if value.ndim == 2:
        return value
    return value.mean(dim=1)


def get_all_candidates(
    text: str, n_gram_range: Union[Tuple[int], List[int]]
) -> List[str]:
    count = CountVectorizer(ngram_range=n_gram_range, stop_words="english").fit([text])
    all_candidates = count.get_feature_names_out()
    return all_candidates


def torch_fast_mode() -> Callable:
    """use `torch.inference_mode()` if torch version is high enough"""
    try:
        return torch.inference_mode()
    except AttributeError:
        return torch.no_grad()
