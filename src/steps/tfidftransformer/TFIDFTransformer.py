import logging
import dill

from src.steps.util.Constants import Constants
from src.steps.util.ModelUtil import ModelUtil

from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)


class TFIDFTransformer(object):

    def __init__(self, action, input_data=""):
        if action == "serve":
            ModelUtil.download_model(
                Constants.TFIDF_VECTORIZER_LOCATION_FILE,
                Constants.TFIDF_VECTORIZER_LOCATION
            )
            self.tfidf_vectorizer = self._load_tfidf_vectorizer(Constants.TFIDF_VECTORIZER_LOCATION)
        elif action == "train":
            ModelUtil.download_to_location(input_data, Constants.TOKENIZED_LOCATION)
            self.tokenized_data = self._load_tokenized_data(Constants.TOKENIZED_LOCATION)

    def _load_tfidf_vectorizer(self, vectorizer_location):
        with open(vectorizer_location, 'rb') as model_file:
            tfidf_vectorizer = dill.load(model_file)
        return tfidf_vectorizer

    def _load_tokenized_data(self, tokenizer_location):
        with open(tokenizer_location, 'rb') as model_file:
            tokenized_data = dill.load(model_file)
        return tokenized_data

    def fit(self):
        x = self.tokenized_data

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            preprocessor=lambda x: x, # We're using cleantext
            tokenizer=lambda x: x, # We're using spacy
            ngram_range=(1, 3)
        )

        self.tfidf_vectorizer.fit(x)

    def transform_vectorizer(self):
        self.tfidf_vectors = self.tfidf_vectorizer.transform(self.tokenized_data)

    def predict(self, X, feature_names):
        logging.info(X)
        tfidf_sparse = self.tfidf_vectorizer.transform(X)
        tfidf_array = tfidf_sparse.toarray()
        logging.info(tfidf_array)
        return tfidf_array
