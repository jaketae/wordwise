import spacy
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from utils import get_all_candidates, squash


class Extractor:
    def __init__(
        self,
        n_gram_range=(1, 2),
        spacy_model="en_core_web_sm",
        bert_model="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    ):
        self.n_gram_range = n_gram_range
        self.nlp = spacy.load(spacy_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

    def generate(self, text, top_k=5):
        text = text.lower()
        candidates = self.get_candidates(text)
        text_embedding = self.get_embedding(text)
        candidate_embeddings = self.get_embedding(candidates)
        distances = cosine_similarity(text_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]
        return keywords

    def get_candidates(self, text):
        nouns = self.get_nouns(text)
        all_candidates = get_all_candidates(text, self.n_gram_range)
        candidates = list(filter(lambda candidate: candidate in nouns, all_candidates))
        return candidates

    def get_nouns(self, text):
        doc = self.nlp(text)
        nouns = set()
        for token in doc:
            if token.pos_ == "NOUN":
                nouns.add(token.text)
        noun_phrases = set(chunk.text.strip() for chunk in doc.noun_chunks)
        return nouns.union(noun_phrases)

    @torch.no_grad()
    def get_embedding(self, source):
        if isinstance(source, str):
            source = [source]
        tokens = self.tokenizer(source, padding=True, return_tensors="pt")
        outputs = self.model(**tokens, return_dict=True)
        embedding = self.parse_outputs(outputs)
        embedding = embedding.detach().numpy()
        return embedding

    def parse_outputs(self, outputs):
        value = None
        outputs_keys = outputs.keys()
        if len(outputs_keys) == 1:
            value = tuple(outputs.values())[0]
        else:
            for key in ["pooler_output", "last_hidden_state"]:
                if key in output_keys:
                    value = outputs[key]
                    break
        if value is None:
            raise RuntimeError("no matching BERT keys found for `outputs`")
        return squash(value)

