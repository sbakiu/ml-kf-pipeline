import spacy
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class Tokenize(object):
    __symbols = set("!$%^&*()_+|~-=`{}[]:\";'<>?,./-")
    nlp = spacy.load('en_core_web_sm', parser=False, entity=False)

    def __init__(self):
        pass

    def predict(self, X, feature_names=[]):
        logging.info(X)
        f = np.vectorize(Tokenize.transform_to_token, otypes=[object])
        x_tokenized = f(X)
        logging.info(x_tokenized)
        return x_tokenized

    @staticmethod
    def transform_to_token(text):
        str_text = str(text)
        doc = Tokenize.nlp(str_text, disable=['parser', 'tagger', 'ner'])
        tokens = []
        logging.info(f"doc = {doc}")
        for token in doc:
            logging.info(f"token = {token}")
            if token.like_url:
                clean_token = "URL"
            else:
                clean_token = token.lemma_.lower().strip()
                if len(clean_token) < 1 or clean_token in \
                        Tokenize.__symbols:
                    continue
            tokens.append(clean_token)
        return tokens
