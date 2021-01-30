import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class Extractor:
    def __init__(self, text, model_name):
        self.text = text.lower()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, top_k=5):
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
        nlp = spacy.load("en_core_web_sm")
        parsed_text = nlp(self.text)
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

    def get_embedding(self, source):
        if isinstance(source, str):
            source = [source]
        tokens = self.tokenizer(source, padding=True, return_tensors="pt")
        embedding = self.model(**tokens)["pooler_output"]
        embedding = embedding.detach().numpy()
        return embedding
