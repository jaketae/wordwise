import spacy
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class Extractor:
    def __init__(
        self,
        spacy_model="en_core_web_sm",
        bert_model="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    ):
        self.nlp = spacy.load(spacy_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

    def generate(self, text, top_k=5):
        self.text = text.lower()
        candidates = self.get_candidates()
        text_embedding = self.get_embedding(self.text)
        candidate_embeddings = self.get_embedding(candidates)
        distances = cosine_similarity(text_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]
        return keywords

    def get_candidates(self):
        noun_phrases = self.get_noun_phrases()
        all_candidates = self.get_all_candidates()
        candidates = list(
            filter(lambda c: c in noun_phrases or len(c.split()) == 1, all_candidates)
        )
        return candidates

    def get_noun_phrases(self):
        parsed_text = self.nlp(self.text)
        noun_phrases = set(chunk.text.strip() for chunk in parsed_text.noun_chunks)
        return noun_phrases

    def get_all_candidates(self):
        n_gram_range = (1, 2)
        stop_words = "english"
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(
            [self.text]
        )
        all_candidates = count.get_feature_names()
        return all_candidates

    @torch.no_grad()
    def get_embedding(self, source):
        if isinstance(source, str):
            source = [source]
        tokens = self.tokenizer(source, padding=True, return_tensors="pt")
        outputs = self.model(**tokens)
        embedding = self.parse_outputs(outputs)
        embedding = embedding.detach().numpy()
        return embedding

    def _squash(self, value):
        if value.ndim == 2:
            return value
        return value.mean(dim=1)

    def parse_outputs(self, outputs):
        value = None
        if isinstance(outputs, dict):
            outputs_keys = outputs.keys()
            if len(outputs_keys) == 1:
                value = tuple(outputs.values())[0]
            else:
                for key in ["pooler_output", "last_hidden_state"]:
                    if key in output_keys:
                        value = outputs[key]
                        break
        else:
            try:
                value = torch.tensor(outputs)
            except:
                raise ValueError(f"Unexpected `outputs` of type {outputs.__class__}")
        return self._squash(value)

