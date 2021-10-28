# WordWise

WordWise is a minimal keyword extraction library that uses quality contextual embeddings generated by Sentence-BERT to extract keywords from blocks of text.

## Installation

WordWise is available on PyPI.

```
pip install wordwise
```

To clone this repository, run

```
git clone https://github.com/jaketae/wordwise.git
```

## Quickstart

At the core of WordWise is the `Extractor` class, which can be configured to generate keywords from some given text. The `Extractor` can be initialized and used out-of-the-box with minimal configuration as follows.

```python
from wordwise import Extractor

extractor = Extractor()
keywords = extractor.generate(text)
```

For advanced users, the `Extractor` class, as well as the `.generate()` method, can be configured in a more granular fashion to induce specific behaviors. For instance, the underlying spaCy backend can be specified.

```python
extractor = Extractor(spacy_model="en_core_web_lg")
```

By default, the `Extractor` will only generate the top 5 most relevant keywords. This behavior can be changed as follows.

```python
# generate 10 keywords
keywords = extractor.generate(text, top_k=10)
```

## Demo

Below is an example text adapted from the [Wikipedia article on supervised learning](https://en.wikipedia.org/wiki/Supervised_learning).

```python
text = """
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
```

The extractor selects the three most relevant keywords from the block of text.

```python
>>> from wordwise import Extractor
>>> extractor = Extractor()
>>> keywords = extractor.generate(text, 3)
>>> print(keywords)
['algorithm', 'learning', 'supervised learning']
```

## How it Works

Using spaCy, the `Extractor` object generates n-gram candidate noun phrases from the provided block of text. By default, it only considers uni-grams or bi-grams, since only rarely are keywords go beyond three words. Then, using a BERT model, it generates contextual embeddings for both the provided text and the n-gram keywords. Using cosine similarity as a distance function, it extracts the `top_k` candidate keywords that are most similar to the embedding of the inputted text.

For a more detailed write-up, please refer to my blog post [here](https://jaketae.github.io/study/keyword-extraction/).

## Credit

WordWise was largely inspired by [KeyBERT](https://github.com/MaartenGr/KeyBERT), a library that similarly uses sentence embeddings for keyword extraction. WordWise also relies on NLP libraries, such as [spaCy](https://spacy.io) and [HuggingFace transformers](https://huggingface.co/transformers/), without which its development would not have been possible.

## License

Released under the MIT License.
