import click
import logging
import pandas as pd

from src.steps.tokenize.Tokenize import Tokenize
from src.steps.util.Config import Config
from src.steps.util.Constants import Constants
from src.steps.util.ModelUtil import ModelUtil

logging.basicConfig(level=logging.INFO)


@click.command()
def run_pipeline():

    csv_encoding = "ISO-8859-1"
    df = pd.read_csv("reddit_train.csv", encoding=csv_encoding)
    features = df.BODY.values
    labels = df.REMOVED.values

    tokenize = Tokenize()

    tokenized_input = tokenize.predict(features)

    ModelUtil.upload_model_with_uuid(
        file_name=Constants.TOKENIZED_FILENAME,
        model=tokenized_input,
        bucket_name=Config.bucket_name,
        model_location_file=Constants.KF_MODEL_OUT_TOKENIZED
    )

    ModelUtil.upload_model_with_uuid(
        file_name=Constants.LABELS_FILENAME,
        model=labels,
        bucket_name=Config.bucket_name,
        model_location_file=Constants.KF_MODEL_OUT_LABELS
    )


if __name__ == "__main__":
    run_pipeline()
