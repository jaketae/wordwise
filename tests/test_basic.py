import unittest

from parameterized import parameterized

from wordwise import Extractor


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.text = """
            Supervised learning is the machine learning task of
            learning a function that maps an input to an output based
            on example input-output pairs.[1] It infers a function
            from labeled training data consisting of a set of
            training examples.[2] In supervised learning, each
            example is a pair consisting of an input object
            (typically a vector) and a desired output value (also
            called the supervisory signal). A supervised learning
            algorithm analyzes the training data and produces an
            inferred function, which can be used for mapping new
            examples. An optimal scenario will allow for the algorithm
            to correctly determine the class labels for unseen
            instances. This requires the learning algorithm to
            generalize from the training data to unseen situations
            in a 'reasonable' way (see inductive bias).
            """

    def generate_inputs():
        for spacy_model in ["en_core_web_sm", "en_core_web_trf"]:
            for bert_model in [
                "bert-base-uncased",
                "sentence-transformers/all-MiniLM-L12-v2",
            ]:
                yield (spacy_model, bert_model)

    @parameterized.expand(generate_inputs())
    def test_extractor(self, spacy_model, bert_model):
        extractor = Extractor(spacy_model=spacy_model, bert_model=bert_model)
        keywords = extractor.generate(self.text, top_k=2)
        self.assertEqual(len(keywords), 2)
