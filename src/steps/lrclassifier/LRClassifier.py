import logging
import dill

from sklearn.linear_model import LogisticRegression

from src.steps.util.Constants import Constants
from src.steps.util.ModelUtil import ModelUtil


class LRClassifier(object):
    def __init__(
            self,
            action,
            labels_data="",
            vectors_data=""):
        if action == "serve":
            ModelUtil.download_model(
                Constants.LR_MODEL_LOCATION_FILE,
                Constants.LR_MODEL_LOCATION
            )
            self.lr_model = self._load_vectors_data(Constants.TFIDF_VECTORIZER_LOCATION)
        elif action == "train":
            ModelUtil.download_to_location(labels_data, Constants.LABELS_LOCATION)
            self.labels_data = self._load_labels_data(Constants.LABELS_LOCATION)

            ModelUtil.download_to_location(vectors_data, Constants.TFIDF_VECTORIZER_LOCATION)
            self.vectors_data = self._load_vectors_data(Constants.TFIDF_VECTORIZER_LOCATION)

    def _load_lr_model(self, model_location):
        with open(model_location, 'rb') as model_file:
            lr_model = dill.load(model_file)
        return lr_model

    def _load_labels_data(self, labels_location):
        with open(labels_location, 'rb') as model_file:
            labels_data = dill.load(model_file)
        return labels_data

    def _load_vectors_data(self, vectors_location):
        with open(vectors_location, 'rb') as model_file:
            vectors_data = dill.load(model_file)
        return vectors_data

    def fit(self):
        self.lr_model = LogisticRegression(
            C=0.1,
            solver='sag')

        self.lr_model.fit(self.vectors_data, self.labels_data)

    def predict(self, X, feature_names):
        logging.info(X)
        prediction = self.lr_model.predict_proba(X)
        logging.info(prediction)
        return prediction
