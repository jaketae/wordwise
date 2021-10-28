import torch
from sklearn.feature_extraction.text import CountVectorizer


def squash(value):
    if not torch.is_tensor(value):
        raise ValueError(f"unexpected `value` of type {value.__class__}")
    if value.ndim == 2:
        return value
    return value.mean(dim=1)


def get_all_candidates(text, n_gram_range):
    count = CountVectorizer(ngram_range=n_gram_range, stop_words="english").fit([text])
    all_candidates = count.get_feature_names_out()
    return all_candidates


def torch_fast_mode():
    """use `torch.inference_mode()` if torch version is high enough"""
    try:
        return torch.inference_mode()
    except AttributeError:
        return torch.no_grad()
