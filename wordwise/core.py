import logging
from typing import Dict, List, Set, Tuple, Union

import spacy
import spacy_transformers  # noqa: F401
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .utils import get_all_candidates, squash, torch_fast_mode

logger = logging.getLogger(__name__)


class Extractor:
    def __init__(
        self,
        n_gram_range: Union[Tuple[int], List[int]] = (1, 2),
        spacy_model: str = "en_core_web_sm",
        bert_model: str = "sentence-transformers/all-MiniLM-L12-v2",
        device: str = "cpu",
    ):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.exception(
                f"Can't find spaCy model {spacy_model}. Have you run `python -m spacy download {spacy_model}`?"
            )
        self.n_gram_range = n_gram_range
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(bert_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

    def generate(self, text: str, top_k: int = 5) -> List[str]:
        candidates = self.get_candidates(text)
        text_embedding = self.get_embedding(text)
        candidate_embeddings = self.get_embedding(candidates)
        distances = F.cosine_similarity(
            text_embedding.unsqueeze(1), candidate_embeddings, dim=-1
        ).squeeze()
        if top_k > len(distances):
            logger.warn(
                f"`top_k` has been adjusted because it is larger than the number of candidates."
            )
            top_k = len(distances)
        _, indicies = torch.topk(distances, k=top_k)
        keywords = [candidates[index] for index in indicies]
        return keywords

    def get_candidates(self, text: str) -> List[str]:
        nouns = self.get_nouns(text)
        all_candidates = get_all_candidates(text, self.n_gram_range)
        candidates = list(filter(lambda candidate: candidate in nouns, all_candidates))
        return candidates

    def get_nouns(self, text: str) -> Set[str]:
        doc = self.nlp(text)
        nouns = set(token.text for token in doc if token.pos_ == "NOUN")
        noun_phrases = set(chunk.text.strip() for chunk in doc.noun_chunks)
        return nouns.union(noun_phrases)

    @torch_fast_mode()
    def get_embedding(self, source: Union[str, List[str]]):
        if isinstance(source, str):
            source = [source]
        tokens = self.tokenizer(
            source,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**tokens, return_dict=True)
        embedding = self.parse_outputs(outputs)
        return embedding

    def parse_outputs(self, outputs: Dict[str, torch.Tensor]):
        value = None
        outputs_keys = outputs.keys()
        if len(outputs_keys) == 1:
            value = tuple(outputs.values())[0]
        else:
            for key in {"pooler_output", "last_hidden_state"}:
                if key in outputs_keys:
                    value = outputs[key]
                    break
        if value is None:
            raise RuntimeError(
                (
                    "No matching BERT keys found from model output. "
                    "Please make sure that the transformer model is BERT-based."
                )
            )
        return squash(value)
