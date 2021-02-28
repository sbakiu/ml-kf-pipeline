class Constants(object):
    TOKENIZED_LOCATION_FILE = "/app/models/tokenized_location.txt"
    TOKENIZED_FILENAME = "tokenized.data"
    TOKENIZED_LOCATION = f"/app/model/{TOKENIZED_FILENAME}"

    LABELS_LOCATION_FILE = "/app/models/labels_location.txt"
    LABELS_FILENAME = "labels.data"
    LABELS_LOCATION = f"/app/model/{LABELS_FILENAME}"

    TFIDF_VECTORIZER_LOCATION_FILE = "/app/models/tfidf_vectorizer_location.txt"
    TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.model"
    TFIDF_VECTORIZER_LOCATION = f"/app/model/{TFIDF_VECTORIZER_FILENAME}"

    TFIDF_VECTORS_LOCATION_FILE = "/app/models/tfidf_vectors_location.txt"
    TFIDF_VECTORS_FILENAME = "tfidf_vectors.data"
    TFIDF_VECTORS_LOCATION = f"/app/model/{TFIDF_VECTORS_FILENAME}"

    LR_MODEL_LOCATION_FILE = "/app/models/lr_model_location.txt"
    LR_MODEL_FILENAME = "lr_text.model"
    LR_MODEL_LOCATION = f"/app/model/{LR_MODEL_FILENAME}"

    KF_MODEL_OUT_TOKENIZED = "/tokenized_location.txt"
    KF_MODEL_OUT_LABELS = "/labels_location.txt"
    KF_MODEL_OUT_TFIDF_VECTORS = "/vectors_location.txt"
    KF_MODEL_OUT_TFIDF_VECTORIZER = "/vectorizer_location.txt"
    KF_MODEL_OUT_LR_MODEL = "/lr_model_location.txt"

