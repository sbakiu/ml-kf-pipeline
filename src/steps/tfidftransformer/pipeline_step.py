import click
import logging

from src.steps.tfidftransformer.TFIDFTransformer import TFIDFTransformer
from src.steps.util.Config import Config
from src.steps.util.Constants import Constants
from src.steps.util.ModelUtil import ModelUtil

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--input-data', default="tokenized.data")
def run_pipeline(input_data):

    transformer = TFIDFTransformer(action="train", input_data=input_data)

    tfidf_vectorizer = transformer.fit()

    ModelUtil.upload_model_with_uuid(
        file_name=Constants.TFIDF_VECTORIZER_FILENAME,
        model=transformer.tfidf_vectorizer,
        bucket_name=Config.bucket_name,
        model_location_file=Constants.KF_MODEL_OUT_TFIDF_VECTORIZER
    )

    transformer.transform_vectorizer()

    ModelUtil.upload_model_with_uuid(
        file_name=Constants.TFIDF_VECTORS_FILENAME,
        model=transformer.tfidf_vectors,
        bucket_name=Config.bucket_name,
        model_location_file=Constants.KF_MODEL_OUT_TFIDF_VECTORS
    )


if __name__ == "__main__":
    run_pipeline()
